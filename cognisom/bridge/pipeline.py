"""
Discovery Pipeline
==================

End-to-end pipeline: generate molecules -> design binders -> simulate effects.
Orchestrates multiple NIMs and feeds results into the cognisom simulation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..nims import MolMIMClient, RFdiffusionClient, ProteinMPNNClient
from .drug_bridge import DrugBridge, DrugCandidate

logger = logging.getLogger(__name__)


def _get_generator(name: str, api_key=None):
    """Return the appropriate molecule generator NIM client."""
    if name == "genmol":
        from ..nims import GenMolClient
        return GenMolClient(api_key=api_key)
    return MolMIMClient(api_key=api_key)


@dataclass
class PipelineResult:
    """Results from a full discovery pipeline run."""
    seed_smiles: str
    target: str
    candidates: List[DrugCandidate] = field(default_factory=list)
    binder_pdb: Optional[str] = None
    binder_sequences: List[str] = field(default_factory=list)
    simulation_results: Optional[Dict] = None


class DiscoveryPipeline:
    """Orchestrates the full drug discovery pipeline.

    Pipeline steps:
    1. MolMIM: Generate candidate small molecules from a seed
    2. DrugBridge: Convert to simulation-ready parameters
    3. (Optional) RFdiffusion: Design protein binder for target
    4. (Optional) ProteinMPNN: Design sequences for the binder
    5. Simulation: Test candidates in digital tissue twin

    Example:
        pipeline = DiscoveryPipeline()
        result = pipeline.run(
            seed_smiles="CC(=O)Oc1ccccc1C(=O)O",  # aspirin scaffold
            target="AR",  # androgen receptor
            num_molecules=20,
        )
        print(f"Top candidate: {result.candidates[0].name}")
        print(f"Kill rate: {result.candidates[0].cancer_kill_rate:.3f}")
    """

    def __init__(self, api_key: Optional[str] = None, molecule_generator: str = "molmim"):
        self.generator = _get_generator(molecule_generator, api_key)
        self.generator_name = molecule_generator
        self.rfdiffusion = RFdiffusionClient(api_key)
        self.proteinmpnn = ProteinMPNNClient(api_key)
        self.bridge = DrugBridge()
        self._api_key = api_key

    def run(self, seed_smiles: str, target: str = "AR",
            num_molecules: int = 20, min_qed: float = 0.5,
            target_pdb: Optional[str] = None,
            target_residues: Optional[str] = None,
            simulation_engine=None) -> PipelineResult:
        """Run the full discovery pipeline.

        Args:
            seed_smiles: Starting molecule SMILES.
            target: Drug target (AR, PI3K, PARP, PD1).
            num_molecules: Number of molecules to generate.
            min_qed: Minimum drug-likeness score.
            target_pdb: (Optional) PDB of target for binder design.
            target_residues: (Optional) Residue range for binder.
            simulation_engine: (Optional) Running simulation to test in.

        Returns:
            PipelineResult with all outputs.
        """
        result = PipelineResult(seed_smiles=seed_smiles, target=target)

        # Step 1: Generate molecules
        logger.info(f"Step 1: Generating {num_molecules} molecules from {seed_smiles} ({self.generator_name})")
        if self.generator_name == "genmol":
            molecules = self.generator.generate(smiles=seed_smiles, num_molecules=num_molecules)
        else:
            molecules = self.generator.generate_for_target(
                seed_smiles, num_molecules, min_qed
            )
        logger.info(f"  Got {len(molecules)} molecules above QED {min_qed}")

        # Step 2: Convert to drug candidates
        logger.info(f"Step 2: Converting to simulation parameters for target {target}")
        result.candidates = self.bridge.convert_molecules(
            molecules, target=target
        )
        logger.info(f"  Top candidate kill rate: {result.candidates[0].cancer_kill_rate:.3f}")

        # Step 3: Design protein binder (if target structure provided)
        if target_pdb and target_residues:
            logger.info(f"Step 3: Designing protein binder for {target_residues}")
            binder = self.rfdiffusion.design_binder(
                target_pdb, target_residues
            )
            result.binder_pdb = binder.pdb_data
            logger.info(f"  Binder generated ({len(binder.pdb_data)} bytes)")

            # Step 4: Design sequences for binder
            logger.info("Step 4: Designing binder sequences")
            sequences = self.proteinmpnn.design_for_binder(binder.pdb_data)
            result.binder_sequences = [s.sequence for s in sequences]
            logger.info(f"  Designed {len(sequences)} sequences")

        # Step 5: Test in simulation (if engine provided)
        if simulation_engine and result.candidates:
            logger.info("Step 5: Testing top candidate in simulation")
            top = result.candidates[0]
            changes = self.bridge.apply_to_simulation(top, simulation_engine)
            result.simulation_results = {
                "candidate": top.name,
                "parameters_changed": changes,
            }

        logger.info("Pipeline complete.")
        return result

    def run_small_molecule_only(self, seed_smiles: str, target: str = "AR",
                                num_molecules: int = 20) -> PipelineResult:
        """Run just the small molecule generation + bridge (no binder design).

        Faster pipeline when you only need small molecule candidates.
        """
        return self.run(
            seed_smiles=seed_smiles,
            target=target,
            num_molecules=num_molecules,
        )

    def run_structure_prediction_pipeline(
        self, sequence: str, method: str = "openfold3"
    ) -> Dict:
        """Predict protein structure: MSA-Search → OpenFold3/Boltz-2.

        Returns dict with structure data and confidence scores.
        """
        from ..nims import MSASearchClient, OpenFold3Client, Boltz2Client

        results: Dict = {"method": method}

        # Step 1: MSA
        try:
            msa = MSASearchClient(api_key=self._api_key)
            msa_result = msa.search(sequence)
            results["msa_sequences"] = msa_result.num_sequences
            alignment = msa_result.alignment
            logger.info(f"MSA: {msa_result.num_sequences} sequences")
        except Exception as exc:
            logger.warning("MSA search failed: %s", exc)
            alignment = None

        # Step 2: Structure prediction
        if method == "boltz2":
            client = Boltz2Client(api_key=self._api_key)
            pred = client.predict_complex(
                polymers=[{"type": "protein", "sequence": sequence}]
            )
            results["structure"] = pred.structure_mmcif
            results["confidence"] = pred.confidence
        else:
            client = OpenFold3Client(api_key=self._api_key)
            pred = client.predict_structure(sequence=sequence, msa=alignment)
            results["structure"] = pred.structure_data
            results["confidence"] = pred.confidence_scores
            results["plddt"] = pred.plddt

        logger.info("Structure prediction complete (%s)", method)
        return results

    def run_mutation_analysis_pipeline(
        self, wt_sequence: str, mutant_sequence: str
    ) -> Dict:
        """Compare wild-type vs mutant using ESM2 embeddings.

        Returns dict with cosine similarity, euclidean distance, interpretation.
        """
        from ..nims import ESM2Client

        esm2 = ESM2Client(api_key=self._api_key)
        impact = esm2.mutation_impact(wt_sequence, mutant_sequence)
        logger.info("Mutation analysis: cosine_sim=%.4f", impact.get("cosine_similarity", 0))
        return impact
