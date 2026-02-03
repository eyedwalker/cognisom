"""
Drug Bridge
===========

Translates NIM-generated drug candidates into simulation parameters.
When MolMIM generates a molecule, this bridge converts its chemical
properties into effects on the cellular simulation (kill rates,
diffusion coefficients, metabolic impacts).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DrugCandidate:
    """A drug candidate with simulation-ready parameters.

    Bridges the gap between NIM output (SMILES, scores) and
    simulation input (kill rates, diffusion, metabolism effects).
    """
    name: str
    smiles: str
    qed_score: float  # Drug-likeness (0-1) from MolMIM

    # Simulation parameters (derived from molecular properties)
    cancer_kill_rate: float = 0.0  # Probability of killing cancer cell per hour
    normal_toxicity: float = 0.0  # Probability of killing normal cell per hour
    diffusion_coefficient: float = 100.0  # um^2/s in tissue
    half_life: float = 4.0  # Hours before degradation
    metabolic_impact: float = 0.0  # Effect on ATP production (-1 to 1)
    immune_modulation: float = 0.0  # Effect on immune activation (-1 to 1)

    # Source tracking
    source_nim: str = ""  # Which NIM generated this
    docking_score: Optional[float] = None  # DiffDock confidence
    target_protein: Optional[str] = None  # What it binds to


class DrugBridge:
    """Converts NIM-generated molecules into simulation drug parameters.

    Uses a simple pharmacological model to estimate simulation effects
    from molecular properties. In production, this would use ADMET
    prediction models.

    Example:
        from cognisom.nims import MolMIMClient
        from cognisom.bridge import DrugBridge

        molmim = MolMIMClient()
        bridge = DrugBridge()

        molecules = molmim.generate("CC(=O)Oc1ccccc1C(=O)O", num_molecules=10)
        candidates = bridge.convert_molecules(molecules, target="AR")
        for c in candidates:
            print(f"{c.name}: kill_rate={c.cancer_kill_rate:.3f}")
    """

    # Known drug target mappings for prostate cancer
    PROSTATE_TARGETS = {
        "AR": {  # Androgen receptor
            "cancer_kill_rate_base": 0.15,
            "immune_modulation": 0.1,
            "description": "Androgen receptor antagonist",
        },
        "PI3K": {  # PI3K/AKT pathway
            "cancer_kill_rate_base": 0.20,
            "immune_modulation": -0.05,
            "description": "PI3K inhibitor",
        },
        "PARP": {  # DNA repair
            "cancer_kill_rate_base": 0.25,
            "normal_toxicity_factor": 0.3,
            "description": "PARP inhibitor (synthetic lethality)",
        },
        "PD1": {  # Immune checkpoint
            "cancer_kill_rate_base": 0.05,
            "immune_modulation": 0.5,
            "description": "Checkpoint inhibitor",
        },
    }

    def convert_molecules(self, molecules, target: str = "AR",
                          source_nim: str = "MolMIM") -> List[DrugCandidate]:
        """Convert MolMIM output molecules to DrugCandidates.

        Args:
            molecules: List of GeneratedMolecule from MolMIM.
            target: Drug target name (AR, PI3K, PARP, PD1).
            source_nim: Name of the generating NIM.

        Returns:
            List of DrugCandidate with simulation parameters.
        """
        target_info = self.PROSTATE_TARGETS.get(target, {})
        base_kill = target_info.get("cancer_kill_rate_base", 0.1)
        base_immune = target_info.get("immune_modulation", 0.0)
        toxicity_factor = target_info.get("normal_toxicity_factor", 0.1)

        candidates = []
        for i, mol in enumerate(molecules):
            # Scale kill rate by QED (drug-likeness predicts efficacy)
            kill_rate = base_kill * mol.score

            # Toxicity inversely related to QED
            toxicity = (1.0 - mol.score) * toxicity_factor

            # Higher QED = better diffusion properties
            diffusion = 50.0 + (mol.score * 100.0)

            # Half-life correlates with drug-likeness
            half_life = 2.0 + (mol.score * 6.0)

            candidate = DrugCandidate(
                name=f"{target}_candidate_{i+1}",
                smiles=mol.smiles,
                qed_score=mol.score,
                cancer_kill_rate=kill_rate,
                normal_toxicity=toxicity,
                diffusion_coefficient=diffusion,
                half_life=half_life,
                metabolic_impact=-0.2 * mol.score,
                immune_modulation=base_immune * mol.score,
                source_nim=source_nim,
                target_protein=target,
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.cancer_kill_rate, reverse=True)
        logger.info(
            f"Converted {len(candidates)} molecules targeting {target}"
        )
        return candidates

    def apply_to_simulation(self, candidate: DrugCandidate,
                            simulation_engine) -> Dict:
        """Apply a drug candidate to a running simulation.

        Modifies simulation parameters to reflect drug effects:
        - Increases cancer cell death rate
        - May affect normal cell viability (toxicity)
        - Modulates immune cell activation
        - Adds drug diffusion to spatial module

        Args:
            candidate: DrugCandidate to apply.
            simulation_engine: Running SimulationEngine instance.

        Returns:
            Dict of parameters that were modified.
        """
        changes = {}

        # Modify cellular module - cancer kill rate
        cellular = simulation_engine.get_module("cellular")
        if cellular:
            cellular.set_parameter("drug_kill_rate", candidate.cancer_kill_rate)
            cellular.set_parameter("drug_toxicity", candidate.normal_toxicity)
            changes["cellular.drug_kill_rate"] = candidate.cancer_kill_rate

        # Modify immune module - activation boost
        immune = simulation_engine.get_module("immune")
        if immune and candidate.immune_modulation > 0:
            immune.set_parameter("activation_boost", candidate.immune_modulation)
            changes["immune.activation_boost"] = candidate.immune_modulation

        # Modify spatial module - drug diffusion field
        spatial = simulation_engine.get_module("spatial")
        if spatial:
            spatial.set_parameter("drug_diffusion", candidate.diffusion_coefficient)
            spatial.set_parameter("drug_half_life", candidate.half_life)
            changes["spatial.drug_diffusion"] = candidate.diffusion_coefficient

        logger.info(
            f"Applied {candidate.name} to simulation: {changes}"
        )
        return changes
