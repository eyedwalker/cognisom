"""NIM-backed tools — protein structure, molecule generation, embeddings, docking."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .tools import Tool, ToolResult

log = logging.getLogger(__name__)

# ── Lazy NIM client loader ───────────────────────────────────────────

_API_KEY = os.getenv("NVIDIA_API_KEY", "")


def _get_client(cls_name: str):
    """Import and instantiate a NIM client on demand."""
    import cognisom.nims as nims
    cls = getattr(nims, cls_name, None)
    if cls is None:
        raise ImportError(f"NIM client {cls_name} not found")
    return cls(api_key=_API_KEY) if _API_KEY else cls()


# ── Structure Prediction ─────────────────────────────────────────────


class StructurePredictionTool(Tool):
    """Predict 3D protein structure from amino acid sequence."""

    name = "structure_prediction"
    description = (
        "Predict the 3D structure of a protein from its amino acid sequence. "
        "Uses MSA-Search + OpenFold3 (or Boltz-2 for complexes)."
    )
    parameters = {
        "sequence": "Amino acid sequence (single-letter code)",
        "method": "'openfold3' or 'boltz2' (default: openfold3)",
    }

    def run(self, *, sequence: str = "", method: str = "openfold3", **kw) -> ToolResult:
        try:
            if not sequence:
                return ToolResult(tool_name=self.name, success=False, error="sequence is required")

            # Optional MSA step
            msa_alignment = None
            try:
                msa = _get_client("MSASearchClient")
                msa_result = msa.search(sequence)
                msa_alignment = msa_result.alignment
                log.info("MSA search returned %d sequences", msa_result.num_sequences)
            except Exception as exc:
                log.warning("MSA search failed (proceeding without): %s", exc)

            # Structure prediction
            if method == "boltz2":
                client = _get_client("Boltz2Client")
                pred = client.predict_complex(
                    polymers=[{"type": "protein", "sequence": sequence}]
                )
                data = {
                    "method": "Boltz-2",
                    "structure_format": "mmcif",
                    "structure_data": pred.structure_mmcif[:500] + "…",
                    "confidence": pred.confidence,
                    "full_structure": pred.structure_mmcif,
                }
            else:
                client = _get_client("OpenFold3Client")
                pred = client.predict_structure(
                    sequence=sequence,
                    msa=msa_alignment,
                )
                data = {
                    "method": "OpenFold3",
                    "structure_format": pred.format,
                    "structure_data": pred.structure_data[:500] + "…",
                    "confidence": pred.confidence_scores,
                    "plddt": pred.plddt,
                    "num_residues": pred.num_residues,
                    "full_structure": pred.structure_data,
                }

            return ToolResult(tool_name=self.name, success=True, data=data)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


# ── Molecule Generation ──────────────────────────────────────────────


class MoleculeGenerationTool(Tool):
    """Generate drug candidate molecules using GenMol."""

    name = "molecule_generation"
    description = (
        "Generate novel drug-like molecules from a seed SMILES or fragment. "
        "Uses NVIDIA GenMol NIM."
    )
    parameters = {
        "smiles": "Seed SMILES string",
        "num_molecules": "Number of molecules to generate (default 10)",
    }

    def run(self, *, smiles: str = "", num_molecules: int = 10, **kw) -> ToolResult:
        try:
            if not smiles:
                return ToolResult(tool_name=self.name, success=False, error="smiles is required")

            client = _get_client("GenMolClient")
            molecules = client.generate(smiles=smiles, num_molecules=num_molecules)

            data = [
                {
                    "smiles": m.smiles,
                    "score": m.score,
                    "qed": getattr(m, "qed", None),
                }
                for m in molecules
            ]
            return ToolResult(tool_name=self.name, success=True, data=data)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


# ── Protein Embedding ────────────────────────────────────────────────


class ProteinEmbeddingTool(Tool):
    """Get protein sequence embeddings and compare sequences."""

    name = "protein_embedding"
    description = (
        "Embed a protein sequence using ESM2-650M. "
        "Optionally compare two sequences via cosine similarity."
    )
    parameters = {
        "sequence": "Primary amino acid sequence",
        "compare_to": "Optional second sequence to compare (cosine similarity)",
    }

    def run(self, *, sequence: str = "", compare_to: str = "", **kw) -> ToolResult:
        try:
            if not sequence:
                return ToolResult(tool_name=self.name, success=False, error="sequence is required")

            client = _get_client("ESM2Client")
            emb = client.embed(sequence)

            data: Dict[str, Any] = {
                "sequence_length": len(sequence),
                "embedding_dimension": emb.dimension,
                "embedding_preview": emb.embedding[:10],
            }

            if compare_to:
                sim = client.similarity(sequence, compare_to)
                data["comparison"] = {
                    "sequence_2_length": len(compare_to),
                    "cosine_similarity": sim,
                }

            return ToolResult(tool_name=self.name, success=True, data=data)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


# ── Mutation Impact ──────────────────────────────────────────────────


class MutationImpactTool(Tool):
    """Score the functional impact of a protein / DNA mutation."""

    name = "mutation_impact"
    description = (
        "Score mutation impact at both protein level (ESM2 embedding shift) "
        "and DNA level (Evo2 log-likelihood). Higher shift = more disruptive."
    )
    parameters = {
        "wild_type": "Wild-type amino acid sequence",
        "mutant": "Mutant amino acid sequence",
        "dna_wt": "Optional wild-type DNA sequence (for Evo2 scoring)",
        "dna_mut": "Optional mutant DNA sequence",
    }

    def run(
        self,
        *,
        wild_type: str = "",
        mutant: str = "",
        dna_wt: str = "",
        dna_mut: str = "",
        **kw,
    ) -> ToolResult:
        try:
            if not wild_type or not mutant:
                return ToolResult(tool_name=self.name, success=False, error="wild_type and mutant are required")

            data: Dict[str, Any] = {}

            # Protein-level (ESM2)
            try:
                esm2 = _get_client("ESM2Client")
                impact = esm2.mutation_impact(wild_type, mutant)
                data["protein_level"] = {
                    "cosine_similarity": impact.get("cosine_similarity"),
                    "euclidean_distance": impact.get("euclidean_distance"),
                    "interpretation": (
                        "High impact" if impact.get("cosine_similarity", 1) < 0.9
                        else "Moderate impact" if impact.get("cosine_similarity", 1) < 0.95
                        else "Low impact"
                    ),
                }
            except Exception as exc:
                data["protein_level"] = {"error": str(exc)}

            # DNA-level (Evo2)
            if dna_wt and dna_mut:
                try:
                    evo2 = _get_client("Evo2Client")
                    # Find mutation position
                    mut_pos = next(
                        (i for i in range(min(len(dna_wt), len(dna_mut))) if dna_wt[i] != dna_mut[i]),
                        None,
                    )
                    if mut_pos is not None:
                        score = evo2.score_mutation(dna_wt, mut_pos, dna_mut[mut_pos])
                        data["dna_level"] = {
                            "position": mut_pos,
                            "wt_base": dna_wt[mut_pos],
                            "mut_base": dna_mut[mut_pos],
                            "log_likelihood_ratio": score.get("log_likelihood_ratio"),
                        }
                except Exception as exc:
                    data["dna_level"] = {"error": str(exc)}

            return ToolResult(tool_name=self.name, success=True, data=data)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


# ── Docking ──────────────────────────────────────────────────────────


class DockingTool(Tool):
    """Dock a small molecule to a protein structure."""

    name = "docking"
    description = (
        "Dock a ligand (SMILES) against a protein structure (PDB text). "
        "Uses DiffDock NIM."
    )
    parameters = {
        "protein_pdb": "Protein structure as PDB-format text",
        "ligand_smiles": "Ligand SMILES string",
        "num_poses": "Number of poses to generate (default 5)",
    }

    def run(self, *, protein_pdb: str = "", ligand_smiles: str = "", num_poses: int = 5, **kw) -> ToolResult:
        try:
            if not protein_pdb or not ligand_smiles:
                return ToolResult(tool_name=self.name, success=False, error="protein_pdb and ligand_smiles are required")

            client = _get_client("DiffDockClient")
            result = client.dock(
                protein_pdb=protein_pdb,
                ligand_smiles=ligand_smiles,
                num_poses=num_poses,
            )

            data = {
                "num_poses": len(result.poses),
                "best_confidence": result.best_confidence,
                "poses": [
                    {
                        "confidence": p.confidence,
                        "coordinates_preview": str(p.ligand_positions[:3]) + "…" if p.ligand_positions else "",
                    }
                    for p in result.poses[:3]
                ],
            }
            return ToolResult(tool_name=self.name, success=True, data=data)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))
