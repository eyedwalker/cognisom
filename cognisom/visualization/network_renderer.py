"""
Cell Interaction Network Renderer
===================================

Renders cell-cell interactions as force-directed network graphs
and contact maps. Shows signaling, killing, and communication
between cells.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Tuple


class InteractionNetworkRenderer:
    """Render cell interaction networks as Plotly figures.

    Example:
        renderer = InteractionNetworkRenderer()
        fig = renderer.render_network(cells, interactions)
        st.plotly_chart(fig)
    """

    # Edge colors by interaction type
    EDGE_COLORS = {
        "paracrine": "#4CAF50",
        "juxtacrine": "#FF9800",
        "killing": "#F44336",
        "exosome": "#9C27B0",
        "chemotaxis": "#2196F3",
        "inhibition": "#607D8B",
    }

    def render_network(
        self,
        cells: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        layout: str = "spatial",
        title: str = "Cell Interaction Network",
    ) -> go.Figure:
        """Render cell interaction network.

        Args:
            cells: List of cell dicts with 'cell_id', 'position', 'cell_type'.
            interactions: List of dicts with 'source', 'target', 'type', 'strength'.
            layout: 'spatial' (use real 3D positions) or 'force' (spring layout).
            title: Plot title.

        Returns:
            Plotly Figure.
        """
        if not cells:
            fig = go.Figure()
            fig.update_layout(title="No cells")
            return fig

        # Build position lookup
        cell_map = {c["cell_id"]: c for c in cells}

        if layout == "spatial":
            return self._render_spatial_network(cell_map, interactions, title)
        else:
            return self._render_2d_network(cell_map, interactions, title)

    def _render_spatial_network(self, cell_map, interactions, title):
        """Render in 3D using actual cell positions."""
        fig = go.Figure()

        # Draw edges first (behind nodes)
        for inter in interactions:
            src = cell_map.get(inter["source"])
            tgt = cell_map.get(inter["target"])
            if not src or not tgt:
                continue

            sp = src["position"]
            tp = tgt["position"]
            itype = inter.get("type", "paracrine")
            strength = inter.get("strength", 0.5)
            color = self.EDGE_COLORS.get(itype, "#999999")

            fig.add_trace(go.Scatter3d(
                x=[sp[0], tp[0]], y=[sp[1], tp[1]], z=[sp[2], tp[2]],
                mode="lines",
                line=dict(color=color, width=max(1, strength * 4)),
                showlegend=False,
                hoverinfo="text",
                text=f"{itype}: {inter['source']} -> {inter['target']} ({strength:.2f})",
            ))

        # Draw nodes
        from .cell_renderer import CELL_TYPE_COLORS
        categories = {}
        for cid, c in cell_map.items():
            ct = c.get("cell_type", "normal")
            if ct not in categories:
                categories[ct] = {"x": [], "y": [], "z": [], "hover": []}
            p = c["position"]
            categories[ct]["x"].append(p[0])
            categories[ct]["y"].append(p[1])
            categories[ct]["z"].append(p[2])
            categories[ct]["hover"].append(f"Cell {cid} ({ct})")

        for ct, data in categories.items():
            fig.add_trace(go.Scatter3d(
                x=data["x"], y=data["y"], z=data["z"],
                mode="markers",
                name=ct,
                marker=dict(
                    size=6,
                    color=CELL_TYPE_COLORS.get(ct, "#9E9E9E"),
                    opacity=0.9,
                    line=dict(width=0.5, color="white"),
                ),
                text=data["hover"],
                hoverinfo="text",
            ))

        # Legend for edge types
        for itype, color in self.EDGE_COLORS.items():
            if any(i.get("type") == itype for i in interactions):
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="lines",
                    name=f"Edge: {itype}",
                    line=dict(color=color, width=3),
                ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (um)", yaxis_title="Y (um)", zaxis_title="Z (um)",
                aspectmode="data",
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    def _render_2d_network(self, cell_map, interactions, title):
        """Render as 2D force-directed graph using spring layout."""
        n = len(cell_map)
        if n == 0:
            return go.Figure()

        ids = list(cell_map.keys())
        id_to_idx = {cid: i for i, cid in enumerate(ids)}

        # Simple spring layout
        pos = self._spring_layout(ids, interactions, id_to_idx)

        fig = go.Figure()

        # Edges
        for inter in interactions:
            si = id_to_idx.get(inter["source"])
            ti = id_to_idx.get(inter["target"])
            if si is None or ti is None:
                continue
            itype = inter.get("type", "paracrine")
            color = self.EDGE_COLORS.get(itype, "#999999")
            strength = inter.get("strength", 0.5)

            fig.add_trace(go.Scatter(
                x=[pos[si][0], pos[ti][0]],
                y=[pos[si][1], pos[ti][1]],
                mode="lines",
                line=dict(color=color, width=max(0.5, strength * 3)),
                showlegend=False,
                hoverinfo="skip",
            ))

        # Nodes
        from .cell_renderer import CELL_TYPE_COLORS
        categories = {}
        for i, cid in enumerate(ids):
            c = cell_map[cid]
            ct = c.get("cell_type", "normal")
            if ct not in categories:
                categories[ct] = {"x": [], "y": [], "text": []}
            categories[ct]["x"].append(pos[i][0])
            categories[ct]["y"].append(pos[i][1])
            categories[ct]["text"].append(f"Cell {cid} ({ct})")

        for ct, data in categories.items():
            fig.add_trace(go.Scatter(
                x=data["x"], y=data["y"],
                mode="markers",
                name=ct,
                marker=dict(
                    size=10,
                    color=CELL_TYPE_COLORS.get(ct, "#9E9E9E"),
                    line=dict(width=1, color="white"),
                ),
                text=data["text"],
                hoverinfo="text",
            ))

        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor="white",
        )
        return fig

    def render_contact_map(
        self,
        cells: List[Dict[str, Any]],
        distance_threshold: float = 30.0,
        title: str = "Cell Contact Map",
    ) -> go.Figure:
        """Render a contact map showing which cells are within interaction range.

        Args:
            cells: List of cell dicts with 'cell_id' and 'position'.
            distance_threshold: Maximum distance (um) for contact.
            title: Plot title.

        Returns:
            Plotly Figure (heatmap).
        """
        n = len(cells)
        if n == 0:
            return go.Figure()

        positions = np.array([c["position"] for c in cells])
        ids = [c["cell_id"] for c in cells]
        types = [c.get("cell_type", "?") for c in cells]

        # Distance matrix
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))

        # Binary contact map
        contact = (dist < distance_threshold).astype(float)
        np.fill_diagonal(contact, 0)

        labels = [f"{cid} ({ct})" for cid, ct in zip(ids, types)]

        fig = go.Figure(data=go.Heatmap(
            z=contact,
            x=labels, y=labels,
            colorscale=[[0, "white"], [1, "#2196F3"]],
            showscale=False,
        ))

        fig.update_layout(
            title=f"{title} (threshold={distance_threshold} um)",
            height=500,
            xaxis=dict(tickangle=45),
        )
        return fig

    @staticmethod
    def _spring_layout(ids, interactions, id_to_idx, iterations=50):
        """Simple spring layout for 2D graph."""
        n = len(ids)
        rng = np.random.default_rng(42)
        pos = rng.uniform(-1, 1, (n, 2))

        # Build adjacency
        adj = np.zeros((n, n))
        for inter in interactions:
            si = id_to_idx.get(inter["source"])
            ti = id_to_idx.get(inter["target"])
            if si is not None and ti is not None:
                adj[si, ti] = 1
                adj[ti, si] = 1

        k = 1.0 / np.sqrt(n + 1)

        for _ in range(iterations):
            disp = np.zeros((n, 2))

            # Repulsive forces
            for i in range(n):
                diff = pos[i] - pos
                dist = np.sqrt(np.sum(diff**2, axis=1))
                dist = np.maximum(dist, 0.01)
                force = diff / dist[:, np.newaxis] * (k**2 / dist[:, np.newaxis])
                force[i] = 0
                disp[i] = np.sum(force, axis=0)

            # Attractive forces
            for inter in interactions:
                si = id_to_idx.get(inter["source"])
                ti = id_to_idx.get(inter["target"])
                if si is None or ti is None:
                    continue
                diff = pos[si] - pos[ti]
                dist = max(np.sqrt(np.sum(diff**2)), 0.01)
                force = diff / dist * (dist**2 / k)
                disp[si] -= force * 0.5
                disp[ti] += force * 0.5

            # Apply with temperature
            temp = 0.1 * (1 - _ / iterations)
            mag = np.sqrt(np.sum(disp**2, axis=1, keepdims=True))
            mag = np.maximum(mag, 0.01)
            pos += disp / mag * min(temp, 0.1)

        return pos

    @staticmethod
    def generate_demo_interactions(cells, n_interactions=30, seed=42):
        """Generate demo interaction data."""
        rng = np.random.default_rng(seed)
        ids = [c["cell_id"] for c in cells]
        interactions = []

        types = ["paracrine", "juxtacrine", "killing", "exosome", "chemotaxis"]

        for _ in range(min(n_interactions, len(ids) * 2)):
            src = rng.choice(ids)
            tgt = rng.choice(ids)
            if src == tgt:
                continue
            interactions.append({
                "source": int(src),
                "target": int(tgt),
                "type": rng.choice(types),
                "strength": float(rng.uniform(0.1, 1.0)),
            })
        return interactions
