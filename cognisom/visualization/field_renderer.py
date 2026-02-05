"""
Spatial Field Volume Renderer
==============================

Renders 3D concentration fields (O2, glucose, cytokines) as
isosurfaces, volume slices, or heatmap cross-sections.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict, Any


class SpatialFieldRenderer:
    """Render 3D concentration fields as interactive Plotly figures.

    Example:
        renderer = SpatialFieldRenderer()
        fig = renderer.render_isosurface(field, name="O2")
        st.plotly_chart(fig)
    """

    def render_isosurface(
        self,
        field: np.ndarray,
        name: str = "Concentration",
        iso_values: Optional[list] = None,
        opacity: float = 0.3,
        colorscale: str = "Viridis",
        resolution_um: float = 10.0,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Render field as 3D isosurface.

        Args:
            field: 3D numpy array (nx, ny, nz).
            name: Field name for labels.
            iso_values: Specific iso-levels. If None, auto-selects 5 levels.
            opacity: Surface opacity.
            colorscale: Plotly colorscale name.
            resolution_um: Physical size per voxel in micrometers.
            title: Plot title.

        Returns:
            Plotly Figure.
        """
        nx, ny, nz = field.shape
        x = np.arange(nx) * resolution_um
        y = np.arange(ny) * resolution_um
        z = np.arange(nz) * resolution_um
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        if iso_values is None:
            vmin, vmax = float(field.min()), float(field.max())
            if vmax > vmin:
                iso_values = np.linspace(vmin + (vmax - vmin) * 0.2,
                                          vmax - (vmax - vmin) * 0.1, 5).tolist()
            else:
                iso_values = [vmin]

        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=field.flatten(),
            isomin=min(iso_values),
            isomax=max(iso_values),
            surface_count=len(iso_values),
            opacity=opacity,
            colorscale=colorscale,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title=name),
        ))

        fig.update_layout(
            title=title or f"{name} Isosurface",
            scene=dict(
                xaxis_title="X (um)",
                yaxis_title="Y (um)",
                zaxis_title="Z (um)",
                aspectmode="data",
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    def render_slices(
        self,
        field: np.ndarray,
        name: str = "Concentration",
        slice_axis: str = "z",
        slice_indices: Optional[list] = None,
        colorscale: str = "Viridis",
        resolution_um: float = 10.0,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Render field as 2D heatmap slices.

        Args:
            field: 3D numpy array.
            name: Field name.
            slice_axis: 'x', 'y', or 'z'.
            slice_indices: Which slices to show. None = middle slice.
            colorscale: Plotly colorscale.
            resolution_um: Physical size per voxel.
            title: Plot title.

        Returns:
            Plotly Figure with slider for slice selection.
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis = axis_map.get(slice_axis, 2)
        n_slices = field.shape[axis]

        if slice_indices is None:
            slice_indices = [n_slices // 2]

        # Axis labels for the 2D plane
        other_axes = [i for i in range(3) if i != axis]
        axis_labels = ["X", "Y", "Z"]

        if len(slice_indices) == 1:
            idx = slice_indices[0]
            slice_2d = np.take(field, idx, axis=axis)

            fig = go.Figure(data=go.Heatmap(
                z=slice_2d.T,
                colorscale=colorscale,
                colorbar=dict(title=name),
            ))
            fig.update_layout(
                title=title or f"{name} — {slice_axis.upper()}={idx * resolution_um:.0f} um",
                xaxis_title=f"{axis_labels[other_axes[0]]} (voxels)",
                yaxis_title=f"{axis_labels[other_axes[1]]} (voxels)",
                height=500,
            )
            return fig

        # Multiple slices — use animation frames
        frames = []
        for idx in range(n_slices):
            slice_2d = np.take(field, idx, axis=axis)
            frames.append(go.Frame(
                data=[go.Heatmap(z=slice_2d.T, colorscale=colorscale)],
                name=str(idx),
            ))

        # Start with middle slice
        mid = n_slices // 2
        init_slice = np.take(field, mid, axis=axis)

        fig = go.Figure(
            data=[go.Heatmap(
                z=init_slice.T,
                colorscale=colorscale,
                colorbar=dict(title=name),
            )],
            frames=frames,
        )

        fig.update_layout(
            title=title or f"{name} Cross-Sections ({slice_axis.upper()} axis)",
            xaxis_title=f"{axis_labels[other_axes[0]]} (voxels)",
            yaxis_title=f"{axis_labels[other_axes[1]]} (voxels)",
            height=500,
            sliders=[dict(
                active=mid,
                steps=[dict(
                    method="animate",
                    args=[[str(i)], dict(mode="immediate", frame=dict(duration=0))],
                    label=f"{i * resolution_um:.0f}",
                ) for i in range(n_slices)],
                currentvalue=dict(prefix=f"{slice_axis.upper()} = ", suffix=" um"),
            )],
        )
        return fig

    def render_multi_field(
        self,
        fields: Dict[str, np.ndarray],
        slice_axis: str = "z",
        slice_index: Optional[int] = None,
        resolution_um: float = 10.0,
    ) -> go.Figure:
        """Render multiple fields side-by-side as 2D slices.

        Args:
            fields: Dict of name -> 3D array.
            slice_axis: Which axis to slice.
            slice_index: Slice position. None = middle.
            resolution_um: Voxel size.

        Returns:
            Plotly Figure with subplots.
        """
        n = len(fields)
        fig = make_subplots(rows=1, cols=n, subplot_titles=list(fields.keys()))

        axis_map = {"x": 0, "y": 1, "z": 2}
        axis = axis_map.get(slice_axis, 2)

        colorscales = ["Viridis", "Inferno", "Plasma", "Cividis", "Turbo"]

        for i, (name, field) in enumerate(fields.items()):
            if slice_index is None:
                idx = field.shape[axis] // 2
            else:
                idx = min(slice_index, field.shape[axis] - 1)

            slice_2d = np.take(field, idx, axis=axis)

            fig.add_trace(
                go.Heatmap(
                    z=slice_2d.T,
                    colorscale=colorscales[i % len(colorscales)],
                    colorbar=dict(title=name, x=0.15 + i * 0.3),
                    showscale=True,
                ),
                row=1, col=i + 1,
            )

        fig.update_layout(
            height=400,
            title=f"Spatial Fields ({slice_axis.upper()} slice)",
        )
        return fig

    @staticmethod
    def generate_demo_fields(grid_size=(50, 50, 30), seed=42):
        """Generate demo concentration fields for testing."""
        rng = np.random.default_rng(seed)
        nx, ny, nz = grid_size

        # O2: high at edges, low in center (tumor hypoxia)
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        oxygen = 0.21 * (0.3 + 0.7 * np.clip(r, 0, 1))
        oxygen += rng.normal(0, 0.005, oxygen.shape)
        oxygen = np.clip(oxygen, 0, 0.21).astype(np.float32)

        # Glucose: similar gradient but different decay
        glucose = 5.0 * (0.2 + 0.8 * np.clip(r, 0, 1))
        glucose += rng.normal(0, 0.1, glucose.shape)
        glucose = np.clip(glucose, 0, 5.0).astype(np.float32)

        # Cytokines: peak near tumor center
        cytokines = 10.0 * np.exp(-3 * r**2)
        cytokines += rng.normal(0, 0.2, cytokines.shape)
        cytokines = np.clip(cytokines, 0, None).astype(np.float32)

        # VEGF: hypoxia-driven, inverse of O2
        vegf = 5.0 * (1 - oxygen / 0.21)
        vegf = np.clip(vegf, 0, None).astype(np.float32)

        return {
            "O2": oxygen,
            "Glucose": glucose,
            "Cytokines": cytokines,
            "VEGF": vegf,
        }
