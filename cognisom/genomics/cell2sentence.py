"""
Cell2Sentence Integration
=========================

Wrapper for the Cell2Sentence-Scale 27B model (vandijklab/C2S-Scale-Gemma-2-27B)
from Yale + Google. Converts single-cell gene expression into natural language
cell state descriptions and predictions.

The model:
- Takes rank-ordered gene "sentences" as input
- Predicts cell type, state, and functional annotations
- Can score T-cell exhaustion, macrophage polarization, etc.
- Based on Gemma-2 27B, fine-tuned on 57M cells from CellxGene/HCA

Hardware: Requires ~24GB VRAM at int8 (fits L40S 48GB with room to spare).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Marker genes for exhaustion scoring (used as fallback when model unavailable)
EXHAUSTION_MARKERS = {
    "inhibitory_receptors": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "CTLA4", "CD244"],
    "exhaustion_tfs": ["TOX", "TOX2", "NR4A1", "NR4A2", "NR4A3"],
    "effector_loss": ["IL2", "TNF", "GZMB", "PRF1", "IFNG"],
    "stemness": ["TCF7", "LEF1", "MYB", "IL7R"],
}

# Macrophage polarization markers
M1_MARKERS = ["TNF", "IL1B", "IL6", "NOS2", "CD80", "CD86", "CXCL10", "CXCL11"]
M2_MARKERS = ["IL10", "MRC1", "CD163", "ARG1", "CCL22", "TGFB1", "VEGFA"]


@dataclass
class CellStatePrediction:
    """Prediction result for a single cell."""
    cell_index: int
    cell_sentence: str
    predicted_cell_type: str
    predicted_state: str
    confidence: float
    exhaustion_score: Optional[float] = None  # 0=effector, 1=exhausted (T cells)
    polarization: Optional[str] = None  # M1/M2 (macrophages)
    polarization_score: Optional[float] = None
    raw_response: str = ""


class Cell2SentenceModel:
    """Wrapper for Cell2Sentence-Scale 27B model.

    Loads the model from HuggingFace with int8 quantization to fit
    on L40S (48GB VRAM). Falls back to a marker-gene-based heuristic
    if the model cannot be loaded (e.g., no GPU, testing locally).

    Example:
        model = Cell2SentenceModel()
        model.load()

        sentence = "GAPDH CD3E TOX PDCD1 HAVCR2 LAG3 TIGIT CTLA4"
        pred = model.predict_cell_state(sentence, cell_index=0)
        print(f"Type: {pred.predicted_cell_type}")
        print(f"Exhaustion: {pred.exhaustion_score:.2f}")
    """

    MODEL_ID = "vandijklab/C2S-Scale-Gemma-2-27B"

    def __init__(self, model_id: Optional[str] = None,
                 quantize: str = "int8",
                 device: str = "auto"):
        """
        Args:
            model_id: HuggingFace model ID (default: C2S-Scale-Gemma-2-27B).
            quantize: Quantization mode — "int8", "int4", or "none".
            device: Device — "auto", "cuda:0", "cpu".
        """
        self.model_id = model_id or self.MODEL_ID
        self.quantize = quantize
        self.device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._fallback_mode = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_fallback(self) -> bool:
        return self._fallback_mode

    def load(self) -> bool:
        """Load the model from HuggingFace.

        Returns True if model loaded, False if falling back to heuristic.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info(f"Loading Cell2Sentence model: {self.model_id}")
            logger.info(f"Quantization: {self.quantize}, Device: {self.device}")

            # Configure quantization
            quant_config = None
            if self.quantize == "int8":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            elif self.quantize == "int4":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map=self.device,
                torch_dtype=torch.float16 if self.quantize == "none" else None,
            )
            self._loaded = True
            self._fallback_mode = False
            logger.info("Cell2Sentence model loaded successfully")
            return True

        except Exception as e:
            logger.warning(
                f"Could not load Cell2Sentence model: {e}. "
                f"Using marker-gene heuristic fallback."
            )
            self._loaded = True
            self._fallback_mode = True
            return False

    def predict_cell_state(self, cell_sentence: str,
                           cell_index: int = 0) -> CellStatePrediction:
        """Predict cell type and state from a gene sentence.

        Args:
            cell_sentence: Space-separated gene names (descending expression).
            cell_index: Index of this cell (for tracking).

        Returns:
            CellStatePrediction with type, state, exhaustion score.
        """
        if not self._loaded:
            self.load()

        if self._fallback_mode:
            return self._predict_heuristic(cell_sentence, cell_index)

        return self._predict_with_model(cell_sentence, cell_index)

    def predict_exhaustion(self, cell_sentence: str) -> float:
        """Score T-cell exhaustion from a gene sentence.

        Returns float 0.0 (fully effector) to 1.0 (fully exhausted).
        """
        pred = self.predict_cell_state(cell_sentence)
        return pred.exhaustion_score if pred.exhaustion_score is not None else 0.0

    def predict_polarization(self, cell_sentence: str) -> Dict:
        """Score macrophage M1/M2 polarization.

        Returns dict with 'polarization' (M1/M2/mixed) and 'score' (-1 to 1).
        """
        pred = self.predict_cell_state(cell_sentence)
        return {
            "polarization": pred.polarization or "unknown",
            "score": pred.polarization_score or 0.0,
        }

    def batch_predict(self, sentences: List[str],
                      cell_indices: Optional[List[int]] = None,
                      ) -> List[CellStatePrediction]:
        """Predict cell states for a batch of sentences.

        Args:
            sentences: List of gene sentences.
            cell_indices: Optional cell indices (defaults to 0, 1, 2, ...).

        Returns:
            List of CellStatePrediction objects.
        """
        if cell_indices is None:
            cell_indices = list(range(len(sentences)))

        predictions = []
        for idx, sentence in zip(cell_indices, sentences):
            pred = self.predict_cell_state(sentence, cell_index=idx)
            predictions.append(pred)

        logger.info(f"Batch predicted {len(predictions)} cell states")
        return predictions

    def _predict_with_model(self, cell_sentence: str,
                            cell_index: int) -> CellStatePrediction:
        """Predict using the loaded transformer model."""
        import torch

        # Build prompt
        prompt = (
            f"The following is a cell represented as a gene expression sentence, "
            f"where genes are ordered by expression level from highest to lowest:\n\n"
            f"{cell_sentence}\n\n"
            f"Based on this gene expression profile, this cell is a"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                top_k=10,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Parse response for cell type and state
        cell_type, state = self._parse_model_response(response)

        # Also compute marker-based scores for exhaustion/polarization
        exhaustion = self._score_exhaustion_markers(cell_sentence)
        polarization, pol_score = self._score_polarization_markers(cell_sentence)

        return CellStatePrediction(
            cell_index=cell_index,
            cell_sentence=cell_sentence[:200],
            predicted_cell_type=cell_type,
            predicted_state=state,
            confidence=0.8,  # Model confidence placeholder
            exhaustion_score=exhaustion,
            polarization=polarization,
            polarization_score=pol_score,
            raw_response=response,
        )

    def _predict_heuristic(self, cell_sentence: str,
                           cell_index: int) -> CellStatePrediction:
        """Fallback: predict cell state from marker gene positions."""
        genes = cell_sentence.upper().split()
        gene_set = set(genes)
        gene_ranks = {g: i for i, g in enumerate(genes)}

        # Determine cell type from top markers
        cell_type = self._classify_cell_type(gene_set, gene_ranks)
        state = "unknown"

        # Exhaustion scoring
        exhaustion = self._score_exhaustion_markers(cell_sentence)
        if exhaustion > 0.6:
            state = "exhausted"
        elif exhaustion > 0.3:
            state = "pre-exhausted"
        elif "CD3E" in gene_set or "CD8A" in gene_set:
            if any(g in gene_set for g in ["GZMB", "PRF1", "IFNG"]):
                state = "effector"
            elif any(g in gene_set for g in ["TCF7", "LEF1", "IL7R"]):
                state = "memory/stem-like"
            else:
                state = "naive"

        # Polarization
        polarization, pol_score = self._score_polarization_markers(cell_sentence)
        if polarization and cell_type == "macrophage":
            state = f"{polarization} polarized"

        return CellStatePrediction(
            cell_index=cell_index,
            cell_sentence=cell_sentence[:200],
            predicted_cell_type=cell_type,
            predicted_state=state,
            confidence=0.5,  # Lower confidence for heuristic
            exhaustion_score=exhaustion,
            polarization=polarization,
            polarization_score=pol_score,
        )

    def _classify_cell_type(self, gene_set: set, gene_ranks: Dict) -> str:
        """Classify cell type from marker gene presence and rank."""
        type_markers = {
            "CD8+ T cell": ["CD8A", "CD8B", "CD3E", "CD3D"],
            "CD4+ T cell": ["CD4", "CD3E", "CD3D"],
            "regulatory T cell": ["FOXP3", "IL2RA", "CTLA4", "CD4"],
            "NK cell": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
            "macrophage": ["CD68", "CD14", "CSF1R", "FCGR3A"],
            "dendritic cell": ["ITGAX", "HLA-DRA", "CD1C", "CLEC9A"],
            "B cell": ["CD19", "MS4A1", "CD79A", "CD79B"],
            "plasma cell": ["SDC1", "MZB1", "XBP1", "JCHAIN"],
            "epithelial": ["EPCAM", "KRT18", "KRT8", "CDH1"],
            "fibroblast": ["COL1A1", "DCN", "FAP", "VIM"],
            "endothelial": ["PECAM1", "VWF", "CDH5", "FLT1"],
            "cancer epithelial": ["EPCAM", "MKI67", "TOP2A", "PCNA"],
        }

        best_type = "unknown"
        best_score = 0

        for cell_type, markers in type_markers.items():
            present = [g for g in markers if g in gene_set]
            # Weight by rank (earlier = higher expression = more weight)
            score = 0
            for g in present:
                rank = gene_ranks.get(g, 999)
                score += max(0, 1.0 - rank / 200.0)  # Higher rank = more score

            if score > best_score:
                best_score = score
                best_type = cell_type

        return best_type

    @staticmethod
    def _score_exhaustion_markers(cell_sentence: str) -> float:
        """Score T-cell exhaustion from marker gene ranks (0-1)."""
        genes = cell_sentence.upper().split()
        gene_ranks = {g: i for i, g in enumerate(genes)}
        n_genes = len(genes) or 1

        score = 0.0
        total_weight = 0.0

        # Inhibitory receptors present and highly ranked → exhaustion
        for gene in EXHAUSTION_MARKERS["inhibitory_receptors"]:
            total_weight += 1.0
            if gene in gene_ranks:
                rank_frac = gene_ranks[gene] / n_genes
                score += max(0, 1.0 - rank_frac)  # Higher rank = more exhaustion signal

        # Exhaustion TFs
        for gene in EXHAUSTION_MARKERS["exhaustion_tfs"]:
            total_weight += 1.5  # Weight TFs more heavily
            if gene in gene_ranks:
                rank_frac = gene_ranks[gene] / n_genes
                score += 1.5 * max(0, 1.0 - rank_frac)

        # Effector genes ABSENT → exhaustion (loss of function)
        for gene in EXHAUSTION_MARKERS["effector_loss"]:
            total_weight += 0.5
            if gene not in gene_ranks:
                score += 0.5  # Absence of effector = exhaustion signal

        # Stemness genes present → LESS exhausted (progenitor exhausted)
        for gene in EXHAUSTION_MARKERS["stemness"]:
            total_weight += 0.5
            if gene in gene_ranks:
                rank_frac = gene_ranks[gene] / n_genes
                score -= 0.5 * max(0, 1.0 - rank_frac)  # Reduce exhaustion

        if total_weight == 0:
            return 0.0
        return max(0.0, min(1.0, score / total_weight))

    @staticmethod
    def _score_polarization_markers(cell_sentence: str) -> tuple:
        """Score macrophage M1 vs M2 polarization.

        Returns (polarization_str, score) where score is -1 (M1) to +1 (M2).
        """
        genes = cell_sentence.upper().split()
        gene_set = set(genes)

        m1_count = sum(1 for g in M1_MARKERS if g in gene_set)
        m2_count = sum(1 for g in M2_MARKERS if g in gene_set)

        total = m1_count + m2_count
        if total == 0:
            return None, 0.0

        # Score: -1 = pure M1, +1 = pure M2
        score = (m2_count - m1_count) / total

        if score < -0.3:
            return "M1", score
        elif score > 0.3:
            return "M2", score
        else:
            return "mixed", score

    @staticmethod
    def _parse_model_response(response: str) -> tuple:
        """Parse natural language cell type/state from model output."""
        response_lower = response.lower().strip()

        # Simple keyword extraction
        cell_type = "unknown"
        state = "unknown"

        type_keywords = {
            "cd8": "CD8+ T cell",
            "cd4": "CD4+ T cell",
            "t cell": "T cell",
            "t-cell": "T cell",
            "regulatory": "regulatory T cell",
            "treg": "regulatory T cell",
            "nk cell": "NK cell",
            "natural killer": "NK cell",
            "macrophage": "macrophage",
            "monocyte": "monocyte",
            "dendritic": "dendritic cell",
            "b cell": "B cell",
            "plasma": "plasma cell",
            "epithelial": "epithelial",
            "fibroblast": "fibroblast",
            "endothelial": "endothelial",
            "cancer": "cancer epithelial",
            "tumor": "cancer epithelial",
        }

        for keyword, ctype in type_keywords.items():
            if keyword in response_lower:
                cell_type = ctype
                break

        state_keywords = {
            "exhaust": "exhausted",
            "dysfunct": "dysfunctional",
            "effector": "effector",
            "memory": "memory",
            "naive": "naive",
            "activated": "activated",
            "resting": "resting",
            "proliferat": "proliferating",
        }

        for keyword, s in state_keywords.items():
            if keyword in response_lower:
                state = s
                break

        return cell_type, state
