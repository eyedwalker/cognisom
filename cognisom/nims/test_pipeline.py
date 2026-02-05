#!/usr/bin/env python3
"""
End-to-End Pipeline Test
========================

Tests the full flow: MolMIM -> DrugBridge -> simulation parameters.
Uses live NVIDIA NIM APIs (requires NVIDIA_API_KEY in .env).

Run:
    cd /Users/davidwalker/CascadeProjects/cognisom
    export $(grep -v '^#' .env | xargs)
    python -m cognisom.nims.test_pipeline
"""

import os
import sys
import json
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_molmim():
    """Test MolMIM molecule generation."""
    from cognisom.nims.molmim import MolMIMClient

    print("=" * 60)
    print("TEST 1: MolMIM - Generate molecules from enzalutamide scaffold")
    print("=" * 60)

    client = MolMIMClient()

    # Enzalutamide (prostate cancer drug) scaffold
    enzalutamide_scaffold = "CC(=O)Nc1ccc(cc1)C#N"

    molecules = client.generate(enzalutamide_scaffold, num_molecules=5)
    print(f"\nGenerated {len(molecules)} molecules:")
    for i, mol in enumerate(molecules):
        print(f"  {i+1}. {mol.smiles}  (QED: {mol.score:.3f})")

    return molecules


def test_drug_bridge(molecules):
    """Test DrugBridge conversion."""
    from cognisom.bridge.drug_bridge import DrugBridge

    print("\n" + "=" * 60)
    print("TEST 2: DrugBridge - Convert to simulation parameters")
    print("=" * 60)

    bridge = DrugBridge()
    candidates = bridge.convert_molecules(molecules, target="AR")

    print(f"\nConverted {len(candidates)} candidates targeting AR:")
    for c in candidates:
        print(f"  {c.name}:")
        print(f"    SMILES: {c.smiles}")
        print(f"    QED: {c.qed_score:.3f}")
        print(f"    Cancer kill rate: {c.cancer_kill_rate:.4f}/hr")
        print(f"    Normal toxicity: {c.normal_toxicity:.4f}/hr")
        print(f"    Diffusion: {c.diffusion_coefficient:.1f} um^2/s")
        print(f"    Half-life: {c.half_life:.1f} hrs")
        print(f"    Immune modulation: {c.immune_modulation:+.3f}")

    return candidates


def test_rfdiffusion():
    """Test RFdiffusion protein binder design."""
    from cognisom.nims.rfdiffusion import RFdiffusionClient

    print("\n" + "=" * 60)
    print("TEST 3: RFdiffusion - Design protein binder")
    print("=" * 60)

    client = RFdiffusionClient()

    # Use a small test protein (hemoglobin fragment)
    import requests
    pdb_text = requests.get("https://files.rcsb.org/download/1R42.pdb").text
    # Extract first 200 ATOM lines
    atom_lines = [l for l in pdb_text.split("\n") if l.startswith("ATOM")][:200]
    pdb_data = "\n".join(atom_lines)

    # Find residue range
    residues = set()
    for line in atom_lines:
        residues.add(int(line[22:26].strip()))
    min_r, max_r = min(residues), max(residues)
    chain = atom_lines[0][21]

    print(f"\nTarget: chain {chain}, residues {min_r}-{max_r}")
    binder = client.design_binder(pdb_data, f"{chain}{min_r}-{max_r}", "70-100")
    print(f"Binder generated: {len(binder.pdb_data)} bytes")
    print(f"First 3 lines:\n{chr(10).join(binder.pdb_data.split(chr(10))[:3])}")

    return binder


def test_proteinmpnn(binder):
    """Test ProteinMPNN sequence design."""
    from cognisom.nims.proteinmpnn import ProteinMPNNClient

    print("\n" + "=" * 60)
    print("TEST 4: ProteinMPNN - Design sequences for binder")
    print("=" * 60)

    client = ProteinMPNNClient()
    sequences = client.design_for_binder(binder.pdb_data, num_sequences=3)

    print(f"\nDesigned {len(sequences)} sequences:")
    for i, seq in enumerate(sequences):
        display_seq = seq.sequence[:50] + "..." if len(seq.sequence) > 50 else seq.sequence
        print(f"  {i+1}. {display_seq}")
        print(f"     Score: {seq.score:.3f}, Recovery: {seq.recovery:.3f}")

    return sequences


def main():
    print("COGNISOM ENGINE - End-to-End Pipeline Test")
    print("Using NVIDIA NIM APIs (live calls)\n")

    # Step 1: Generate molecules
    molecules = test_molmim()

    # Step 2: Convert to simulation parameters
    candidates = test_drug_bridge(molecules)

    # Step 3: Design protein binder
    binder = test_rfdiffusion()

    # Step 4: Design sequences
    sequences = test_proteinmpnn(binder)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Molecules generated: {len(molecules)}")
    print(f"  Drug candidates: {len(candidates)}")
    print(f"  Top candidate kill rate: {candidates[0].cancer_kill_rate:.4f}/hr")
    print(f"  Binder designed: {len(binder.pdb_data)} bytes")
    print(f"  Binder sequences: {len(sequences)}")
    print("\nAll NVIDIA NIM APIs working. Pipeline operational.")


if __name__ == "__main__":
    main()
