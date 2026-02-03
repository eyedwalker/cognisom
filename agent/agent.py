"""Research Agent orchestrator — predefined workflows and interactive queries."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tools import ToolRegistry, ToolResult
from .db_tools import (
    NCBIGeneTool,
    UniProtTool,
    PDBSearchTool,
    CBioPortalTool,
    PubMedSearchTool,
)
from .nim_tools import (
    StructurePredictionTool,
    MoleculeGenerationTool,
    ProteinEmbeddingTool,
    MutationImpactTool,
    DockingTool,
)

log = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """One step in a multi-tool workflow."""
    tool_name: str
    kwargs: Dict[str, Any]
    label: str = ""
    depends_on: Optional[str] = None  # key in results dict to pull data from


@dataclass
class WorkflowResult:
    """Collected results from a multi-step workflow."""
    workflow_name: str
    steps_completed: int = 0
    steps_total: int = 0
    results: Dict[str, ToolResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def summary(self) -> str:
        lines = [f"Workflow: {self.workflow_name} ({self.steps_completed}/{self.steps_total} steps)"]
        for key, r in self.results.items():
            lines.append(f"  {key}: {r.summary}")
        if self.errors:
            lines.append(f"  Errors: {'; '.join(self.errors)}")
        return "\n".join(lines)


class ResearchAgent:
    """Orchestrates multi-tool research workflows.

    Usage::

        agent = ResearchAgent()
        result = agent.investigate_gene("TP53")
        result = agent.mutation_analysis("TP53", wt_seq, mut_seq)
        result = agent.drug_target_exploration("AR", "CC1=CC=CC=C1")

    You can also run individual tools::

        result = agent.run_tool("ncbi_gene", query="BRCA1")
    """

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        self.registry = ToolRegistry()
        self._register_all_tools()

    def _register_all_tools(self) -> None:
        # Database tools
        self.registry.register(NCBIGeneTool())
        self.registry.register(UniProtTool())
        self.registry.register(PDBSearchTool())
        self.registry.register(CBioPortalTool())
        self.registry.register(PubMedSearchTool())
        # NIM-backed tools
        self.registry.register(StructurePredictionTool())
        self.registry.register(MoleculeGenerationTool())
        self.registry.register(ProteinEmbeddingTool())
        self.registry.register(MutationImpactTool())
        self.registry.register(DockingTool())

    def run_tool(self, name: str, **kwargs) -> ToolResult:
        """Run a single tool by name."""
        return self.registry.run(name, **kwargs)

    def list_tools(self) -> List[Dict[str, str]]:
        return self.registry.list_tools()

    # ── Predefined Workflows ────────────────────────────────────────

    def investigate_gene(
        self,
        gene: str,
        study: str = "prad_tcga",
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> WorkflowResult:
        """Full gene investigation: NCBI → UniProt → PDB → cBioPortal → PubMed.

        Parameters
        ----------
        gene : str
            Gene symbol (e.g. ``"TP53"``).
        study : str
            cBioPortal study ID (default: prostate adenocarcinoma).
        progress_cb : callable, optional
            ``fn(step_label, current, total)`` called after each step.
        """
        wf = WorkflowResult(workflow_name=f"Gene Investigation: {gene}", steps_total=5)

        steps = [
            ("gene_info", "ncbi_gene", {"query": gene}, "Looking up gene in NCBI"),
            ("protein_info", "uniprot", {"query": gene}, "Fetching protein from UniProt"),
            ("structures", "pdb_search", {"query": gene}, "Searching PDB structures"),
            ("cancer_mutations", "cbioportal", {"gene": gene, "study": study}, "Querying cancer mutations"),
            ("literature", "pubmed_search", {"query": f"{gene} prostate cancer", "max_results": 5}, "Searching PubMed"),
        ]

        for i, (key, tool_name, kwargs, label) in enumerate(steps):
            try:
                if progress_cb:
                    progress_cb(label, i, len(steps))
                result = self.registry.run(tool_name, **kwargs)
                wf.results[key] = result
                wf.steps_completed += 1
                if not result.success:
                    wf.errors.append(f"{key}: {result.error}")
            except Exception as exc:
                wf.errors.append(f"{key}: {exc}")

        if progress_cb:
            progress_cb("Done", len(steps), len(steps))
        return wf

    def mutation_analysis(
        self,
        gene: str,
        wt_sequence: str,
        mutant_sequence: str,
        dna_wt: str = "",
        dna_mut: str = "",
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> WorkflowResult:
        """Analyse a mutation: NCBI → ESM2 embedding shift → Evo2 DNA scoring → Structure comparison.

        Parameters
        ----------
        gene : str
            Gene symbol for context.
        wt_sequence : str
            Wild-type amino acid sequence.
        mutant_sequence : str
            Mutant amino acid sequence.
        dna_wt, dna_mut : str
            Optional DNA sequences for Evo2 scoring.
        """
        wf = WorkflowResult(workflow_name=f"Mutation Analysis: {gene}", steps_total=4)

        steps: list = [
            ("gene_info", "ncbi_gene", {"query": gene}, "Looking up gene"),
            ("embedding_comparison", "protein_embedding", {
                "sequence": wt_sequence,
                "compare_to": mutant_sequence,
            }, "Comparing WT vs mutant embeddings"),
            ("mutation_impact", "mutation_impact", {
                "wild_type": wt_sequence,
                "mutant": mutant_sequence,
                "dna_wt": dna_wt,
                "dna_mut": dna_mut,
            }, "Scoring mutation impact"),
            ("literature", "pubmed_search", {
                "query": f"{gene} mutation prostate cancer",
                "max_results": 3,
            }, "Searching mutation literature"),
        ]

        for i, (key, tool_name, kwargs, label) in enumerate(steps):
            try:
                if progress_cb:
                    progress_cb(label, i, len(steps))
                result = self.registry.run(tool_name, **kwargs)
                wf.results[key] = result
                wf.steps_completed += 1
                if not result.success:
                    wf.errors.append(f"{key}: {result.error}")
            except Exception as exc:
                wf.errors.append(f"{key}: {exc}")

        if progress_cb:
            progress_cb("Done", len(steps), len(steps))
        return wf

    def drug_target_exploration(
        self,
        gene: str,
        seed_smiles: str = "",
        num_molecules: int = 10,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> WorkflowResult:
        """Drug target pipeline: UniProt → PDB/Structure → GenMol → DiffDock.

        Parameters
        ----------
        gene : str
            Target gene symbol.
        seed_smiles : str
            Seed SMILES for molecule generation.
        num_molecules : int
            Number of candidate molecules to generate.
        """
        wf = WorkflowResult(workflow_name=f"Drug Target: {gene}", steps_total=4)

        steps_def: list = [
            ("protein_info", "uniprot", {"query": gene}, "Fetching target protein"),
            ("structures", "pdb_search", {"query": gene, "max_results": 1}, "Finding target structure"),
        ]

        # Only add generation + docking if seed_smiles provided
        if seed_smiles:
            steps_def.append(
                ("generated_molecules", "molecule_generation", {
                    "smiles": seed_smiles,
                    "num_molecules": num_molecules,
                }, "Generating drug candidates")
            )
        wf.steps_total = len(steps_def)

        for i, (key, tool_name, kwargs, label) in enumerate(steps_def):
            try:
                if progress_cb:
                    progress_cb(label, i, len(steps_def))
                result = self.registry.run(tool_name, **kwargs)
                wf.results[key] = result
                wf.steps_completed += 1
                if not result.success:
                    wf.errors.append(f"{key}: {result.error}")
            except Exception as exc:
                wf.errors.append(f"{key}: {exc}")

        # Attempt docking if we have both structure and molecule
        if (
            seed_smiles
            and "structures" in wf.results
            and wf.results["structures"].success
            and wf.results["structures"].data
        ):
            try:
                if progress_cb:
                    progress_cb("Docking best candidate", wf.steps_completed, wf.steps_total + 1)
                wf.steps_total += 1

                # Fetch actual PDB text for the top structure
                pdb_id = None
                struct_data = wf.results["structures"].data
                if isinstance(struct_data, list) and struct_data:
                    pdb_id = struct_data[0].get("pdb_id")

                if pdb_id:
                    import urllib.request
                    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                    req = urllib.request.Request(pdb_url, headers={"User-Agent": "Cognisom/1.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        pdb_text = resp.read().decode()

                    # Use first generated molecule or seed
                    dock_smiles = seed_smiles
                    if "generated_molecules" in wf.results and wf.results["generated_molecules"].success:
                        gen_data = wf.results["generated_molecules"].data
                        if isinstance(gen_data, list) and gen_data:
                            dock_smiles = gen_data[0].get("smiles", seed_smiles)

                    dock_result = self.registry.run(
                        "docking",
                        protein_pdb=pdb_text,
                        ligand_smiles=dock_smiles,
                    )
                    wf.results["docking"] = dock_result
                    wf.steps_completed += 1
            except Exception as exc:
                wf.errors.append(f"docking: {exc}")

        if progress_cb:
            progress_cb("Done", wf.steps_completed, wf.steps_total)
        return wf
