"""
3D Cell Population Renderer
============================

Renders cell populations as interactive 3D scatter plots.
Each cell is a sphere colored by type, phase, or metabolic state.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Tuple


# Color palettes
CELL_TYPE_COLORS = {
    "normal": "#4CAF50",
    "cancer": "#F44336",
    "immune": "#2196F3",
    "t_cell": "#2196F3",
    "nk_cell": "#9C27B0",
    "macrophage": "#FF9800",
    "fibroblast": "#795548",
    "endothelial": "#00BCD4",
    "stem": "#E91E63",
    "dead": "#9E9E9E",
}

PHASE_COLORS = {
    "G1": "#4CAF50",
    "S": "#2196F3",
    "G2": "#FF9800",
    "M": "#F44336",
    "G0": "#9E9E9E",
}


class CellPopulationRenderer:
    """Render cell populations as interactive 3D Plotly figures.

    Example:
        renderer = CellPopulationRenderer()
        fig = renderer.render(cells, color_by="type")
        st.plotly_chart(fig)
    """

    def render(
        self,
        cells: List[Dict[str, Any]],
        color_by: str = "type",
        size_by: str = "fixed",
        show_dead: bool = False,
        highlight_mutations: bool = False,
        camera: Optional[Dict] = None,
        title: str = "Cell Population",
    ) -> go.Figure:
        """Render cells as 3D scatter plot.

        Args:
            cells: List of cell dicts with keys: position, cell_type, phase,
                   alive, oxygen, glucose, atp, mhc1_expression, mutations.
            color_by: 'type', 'phase', 'oxygen', 'glucose', 'atp', 'mhc1'.
            size_by: 'fixed', 'volume', 'atp'.
            show_dead: Include dead cells (gray, transparent).
            highlight_mutations: Add markers for mutated cells.
            camera: Custom camera position dict.
            title: Plot title.

        Returns:
            Plotly Figure.
        """
        if not cells:
            fig = go.Figure()
            fig.update_layout(title="No cells to display")
            return fig

        # Filter
        active = [c for c in cells if c.get("alive", True) or show_dead]
        if not active:
            fig = go.Figure()
            fig.update_layout(title="No cells to display")
            return fig

        # Extract positions
        positions = np.array([c["position"] for c in active])
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        # Colors
        colors, colorbar_title = self._get_colors(active, color_by)

        # Sizes
        sizes = self._get_sizes(active, size_by)

        # Hover text
        hover = [self._hover_text(c) for c in active]

        fig = go.Figure()

        if isinstance(colors[0], str):
            # Categorical colors — group by category for legend
            categories = {}
            for i, c in enumerate(active):
                cat = c.get("cell_type", "unknown") if color_by == "type" else c.get("phase", "?")
                if cat not in categories:
                    categories[cat] = {"x": [], "y": [], "z": [], "sizes": [], "hover": [], "color": colors[i]}
                categories[cat]["x"].append(x[i])
                categories[cat]["y"].append(y[i])
                categories[cat]["z"].append(z[i])
                categories[cat]["sizes"].append(sizes[i])
                categories[cat]["hover"].append(hover[i])

            for cat, data in categories.items():
                fig.add_trace(go.Scatter3d(
                    x=data["x"], y=data["y"], z=data["z"],
                    mode="markers",
                    name=cat,
                    marker=dict(
                        size=data["sizes"],
                        color=data["color"],
                        opacity=0.8,
                        line=dict(width=0.5, color="white"),
                    ),
                    text=data["hover"],
                    hoverinfo="text",
                ))
        else:
            # Continuous colorscale
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title=colorbar_title),
                    line=dict(width=0.5, color="white"),
                ),
                text=hover,
                hoverinfo="text",
            ))

        # Highlight mutated cells
        if highlight_mutations:
            mutated = [c for c in active if c.get("mutations")]
            if mutated:
                m_pos = np.array([c["position"] for c in mutated])
                fig.add_trace(go.Scatter3d(
                    x=m_pos[:, 0], y=m_pos[:, 1], z=m_pos[:, 2],
                    mode="markers",
                    name="Mutated",
                    marker=dict(
                        size=8, color="rgba(0,0,0,0)",
                        line=dict(width=2, color="#FFD700"),
                        symbol="diamond",
                    ),
                    text=[", ".join(c.get("mutations", [])) for c in mutated],
                    hoverinfo="text",
                ))

        # Layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (um)",
                yaxis_title="Y (um)",
                zaxis_title="Z (um)",
                aspectmode="data",
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        if camera:
            fig.update_layout(scene_camera=camera)

        return fig

    def render_from_engine(self, engine_state: Dict[str, Any], module: str = "cellular",
                           **kwargs) -> go.Figure:
        """Render directly from SimulationEngine.get_state() output."""
        module_state = engine_state.get(module, {})
        cells_raw = module_state.get("cells", [])

        if isinstance(cells_raw, dict):
            # cells is a dict of cell_id -> cell_state
            cells = []
            for cid, cs in cells_raw.items():
                if hasattr(cs, "__dict__"):
                    d = cs.__dict__.copy()
                    if hasattr(d.get("position"), "tolist"):
                        d["position"] = d["position"].tolist()
                    cells.append(d)
                elif isinstance(cs, dict):
                    cells.append(cs)
        elif isinstance(cells_raw, list):
            cells = cells_raw
        else:
            cells = []

        return self.render(cells, **kwargs)

    def _get_colors(self, cells, color_by):
        if color_by == "type":
            return [CELL_TYPE_COLORS.get(c.get("cell_type", "normal"), "#9E9E9E")
                    for c in cells], "Cell Type"
        elif color_by == "phase":
            return [PHASE_COLORS.get(c.get("phase", "G1"), "#9E9E9E")
                    for c in cells], "Phase"
        elif color_by == "oxygen":
            return [c.get("oxygen", 0.21) for c in cells], "O2"
        elif color_by == "glucose":
            return [c.get("glucose", 5.0) for c in cells], "Glucose (mM)"
        elif color_by == "atp":
            return [c.get("atp", 1000) for c in cells], "ATP"
        elif color_by == "mhc1":
            return [c.get("mhc1_expression", 1.0) for c in cells], "MHC-I"
        else:
            return ["#4CAF50"] * len(cells), ""

    def _get_sizes(self, cells, size_by):
        if size_by == "volume":
            # Larger cells get bigger markers
            return [max(3, min(12, c.get("volume", 1.0) * 5)) for c in cells]
        elif size_by == "atp":
            atps = [c.get("atp", 1000) for c in cells]
            max_atp = max(atps) if atps else 1
            return [max(3, v / max(max_atp, 1) * 10) for v in atps]
        return [6] * len(cells)

    @staticmethod
    def _hover_text(cell):
        lines = [
            f"Type: {cell.get('cell_type', '?')}",
            f"Phase: {cell.get('phase', '?')}",
            f"O2: {cell.get('oxygen', 0):.3f}",
            f"Glucose: {cell.get('glucose', 0):.2f}",
            f"ATP: {cell.get('atp', 0):.0f}",
        ]
        if cell.get("mutations"):
            lines.append(f"Mutations: {', '.join(cell['mutations'])}")
        if cell.get("mhc1_expression") is not None:
            lines.append(f"MHC-I: {cell['mhc1_expression']:.2f}")
        return "<br>".join(lines)

    @staticmethod
    def generate_demo_cells(n_normal=60, n_cancer=20, n_immune=10, seed=42):
        """Generate demo cell data for testing the renderer."""
        rng = np.random.default_rng(seed)
        cells = []

        # Normal cells — ring
        for i in range(n_normal):
            angle = 2 * np.pi * i / n_normal
            r = 50
            cells.append({
                "cell_id": i,
                "position": [100 + r * np.cos(angle) + rng.normal(0, 3),
                             100 + r * np.sin(angle) + rng.normal(0, 3),
                             50 + rng.normal(0, 5)],
                "cell_type": "normal",
                "phase": rng.choice(["G1", "S", "G2", "M"], p=[0.5, 0.2, 0.2, 0.1]),
                "alive": True,
                "oxygen": 0.18 + rng.normal(0, 0.02),
                "glucose": 4.5 + rng.normal(0, 0.3),
                "atp": 900 + rng.normal(0, 100),
                "mhc1_expression": 0.9 + rng.normal(0, 0.05),
                "mutations": [],
            })

        # Cancer cells — cluster
        for i in range(n_cancer):
            cells.append({
                "cell_id": n_normal + i,
                "position": [120 + rng.normal(0, 10),
                             120 + rng.normal(0, 10),
                             50 + rng.normal(0, 5)],
                "cell_type": "cancer",
                "phase": rng.choice(["G1", "S", "G2", "M"], p=[0.3, 0.3, 0.2, 0.2]),
                "alive": True,
                "oxygen": 0.08 + rng.normal(0, 0.02),
                "glucose": 2.0 + rng.normal(0, 0.5),
                "atp": 600 + rng.normal(0, 100),
                "mhc1_expression": 0.3 + rng.normal(0, 0.1),
                "mutations": [["PTEN_loss"], ["TP53_R175H"], ["AR_amplification"], []][
                    rng.choice(4, p=[0.3, 0.2, 0.2, 0.3])
                ],
            })

        # Immune cells — scattered
        for i in range(n_immune):
            cells.append({
                "cell_id": n_normal + n_cancer + i,
                "position": [100 + rng.normal(0, 40),
                             100 + rng.normal(0, 40),
                             50 + rng.normal(0, 10)],
                "cell_type": rng.choice(["t_cell", "nk_cell", "macrophage"]),
                "phase": "G0",
                "alive": True,
                "oxygen": 0.15 + rng.normal(0, 0.03),
                "glucose": 3.5 + rng.normal(0, 0.5),
                "atp": 800 + rng.normal(0, 100),
                "mhc1_expression": 1.0,
                "mutations": [],
            })

        return cells
