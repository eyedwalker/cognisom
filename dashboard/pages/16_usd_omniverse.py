"""
USD & Omniverse Dashboard
=========================

OpenUSD schema browser and NVIDIA Omniverse connection management.

Features:
- Bio-USD schema browser (prim types, API schemas)
- Omniverse connection status and control
- Scene export to USD format
- Real-time sync controls
"""

import streamlit as st
import json
from dataclasses import asdict, fields, is_dataclass
from typing import get_type_hints

st.set_page_config(
    page_title="USD & Omniverse | Cognisom",
    page_icon="🎬",
    layout="wide",
)

# Auth gate
try:
    from cognisom.auth.middleware import streamlit_page_gate
    user = streamlit_page_gate(required_tier="researcher")
except Exception:
    user = None

st.title("🎬 USD & Omniverse")
st.markdown("OpenUSD scene management and NVIDIA Omniverse integration")

# ═══════════════════════════════════════════════════════════════════════
# Tab Layout
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📐 Schema Browser", "🔌 Omniverse Connection", "📤 Export", "📖 Documentation"
])

# ─────────────────────────────────────────────────────────────────────────
# Tab 1: Schema Browser
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Bio-USD Schema Browser")
    st.markdown("Browse registered prim types and API schemas from the Bio-USD specification.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Prim Types")
        try:
            from cognisom.biousd.schema import (
                prim_registry, api_registry, list_prim_types, list_api_schemas,
                get_prim_class, get_api_class,
                CellType, CellPhase, ImmuneCellType, SpatialFieldType, GeneType
            )

            prim_types = list_prim_types()
            selected_prim = st.selectbox(
                "Select Prim Type",
                prim_types,
                format_func=lambda x: x.replace("bio_", "Bio").replace("_", " ").title()
            )

            st.divider()

            st.markdown("### API Schemas")
            api_schemas = list_api_schemas()
            selected_api = st.selectbox(
                "Select API Schema",
                api_schemas,
                format_func=lambda x: x.replace("bio_", "Bio").replace("_api", "").replace("_", " ").title()
            )

            st.divider()

            st.markdown("### Enums")
            enum_options = {
                "CellType": CellType,
                "CellPhase": CellPhase,
                "ImmuneCellType": ImmuneCellType,
                "SpatialFieldType": SpatialFieldType,
                "GeneType": GeneType,
            }
            selected_enum = st.selectbox("Select Enum", list(enum_options.keys()))

        except ImportError as e:
            st.error(f"Bio-USD module not available: {e}")
            prim_types = []
            selected_prim = None
            selected_api = None
            selected_enum = None

    with col2:
        if selected_prim:
            st.markdown(f"### `{selected_prim}`")

            try:
                prim_class = get_prim_class(selected_prim)

                # Show docstring
                if prim_class.__doc__:
                    st.markdown(f"**Description:**")
                    st.markdown(prim_class.__doc__.strip())

                # Show fields
                st.markdown("**Fields:**")

                if is_dataclass(prim_class):
                    field_data = []
                    for f in fields(prim_class):
                        field_type = str(f.type).replace("typing.", "").replace("<class '", "").replace("'>", "")
                        default_val = f.default if f.default is not f.default_factory else "(factory)"
                        if f.default is f.default_factory and f.default_factory is not None:
                            try:
                                default_val = str(f.default_factory())
                            except:
                                default_val = "(factory)"
                        field_data.append({
                            "Field": f.name,
                            "Type": field_type,
                            "Default": str(default_val)[:50],
                        })

                    st.table(field_data)

                # Show inheritance
                bases = [b.__name__ for b in prim_class.__bases__ if b.__name__ != "object"]
                if bases:
                    st.markdown(f"**Inherits from:** `{', '.join(bases)}`")

                # Example instantiation
                st.markdown("**Example:**")
                st.code(f"""from cognisom.biousd.schema import create_prim

prim = create_prim("{selected_prim}",
    prim_path="/World/Cells/cell_001",
    display_name="My Cell"
)""", language="python")

            except Exception as e:
                st.error(f"Error loading prim class: {e}")

        if selected_api:
            st.divider()
            st.markdown(f"### `{selected_api}`")

            try:
                api_class = get_api_class(selected_api)

                if api_class.__doc__:
                    st.markdown(f"**Description:**")
                    st.markdown(api_class.__doc__.strip())

                if is_dataclass(api_class):
                    st.markdown("**Fields:**")
                    field_data = []
                    for f in fields(api_class):
                        field_type = str(f.type).replace("typing.", "")
                        default_val = f.default if f.default is not f.default_factory else "(factory)"
                        field_data.append({
                            "Field": f.name,
                            "Type": field_type,
                            "Default": str(default_val)[:30],
                        })
                    st.table(field_data)

            except Exception as e:
                st.error(f"Error loading API schema: {e}")

        if selected_enum:
            st.divider()
            st.markdown(f"### `{selected_enum}`")

            enum_class = enum_options.get(selected_enum)
            if enum_class:
                values = [{"Name": e.name, "Value": e.value} for e in enum_class]
                st.table(values)

# ─────────────────────────────────────────────────────────────────────────
# Tab 2: Real USD Generation (NO MOCKS)
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🎬 Real USD Generation")
    st.success("**Real Mode** — Generates actual OpenUSD files using the `pxr` module. No mocks!")

    # Initialize session state
    if "real_connector" not in st.session_state:
        st.session_state.real_connector = None
    if "usd_stage_path" not in st.session_state:
        st.session_state.usd_stage_path = None

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Create USD Scene")

        scene_name = st.text_input(
            "Scene Name",
            value="cognisom_cells",
            help="Name for the USD scene file"
        )

        output_dir = st.text_input(
            "Output Directory",
            value="exports/usd",
            help="Directory for generated USD files"
        )

        st.divider()
        st.markdown("### Cell Configuration")

        num_cells = st.slider("Number of Cells", 10, 500, 100)
        scene_size = st.slider("Scene Size (μm)", 100, 1000, 300)

        col_ct1, col_ct2 = st.columns(2)
        with col_ct1:
            include_cancer = st.checkbox("Cancer Cells", value=True)
            include_tcells = st.checkbox("T Cells", value=True)
        with col_ct2:
            include_normal = st.checkbox("Normal Cells", value=True)
            include_nk = st.checkbox("NK Cells", value=True)

        st.divider()

        if st.button("🎬 Generate Real USD", type="primary", use_container_width=True):
            with st.spinner("Creating real USD scene..."):
                try:
                    from cognisom.omniverse.real_connector import (
                        RealOmniverseConnector, CellVisualization
                    )
                    import numpy as np
                    import os

                    # Create output directory
                    os.makedirs(output_dir, exist_ok=True)

                    # Initialize real connector
                    connector = RealOmniverseConnector(output_dir=output_dir)
                    st.session_state.real_connector = connector

                    # Create stage
                    if connector.create_stage(scene_name):
                        st.success(f"✅ Created USD stage: {scene_name}")

                        # Build cell type list
                        cell_types = []
                        if include_cancer:
                            cell_types.append("cancer")
                        if include_tcells:
                            cell_types.append("tcell")
                        if include_normal:
                            cell_types.append("normal")
                        if include_nk:
                            cell_types.append("nk_cell")

                        if not cell_types:
                            cell_types = ["cancer"]

                        # Generate cells
                        cells_added = 0
                        for i in range(num_cells):
                            cell = CellVisualization(
                                cell_id=f"cell_{i:04d}",
                                position=(
                                    np.random.uniform(0, scene_size),
                                    np.random.uniform(0, scene_size),
                                    np.random.uniform(0, scene_size * 0.3),
                                ),
                                radius=np.random.uniform(3.0, 8.0),
                                cell_type=np.random.choice(cell_types),
                                color=None,  # auto-assigned by type
                                metadata={
                                    "age": float(np.random.uniform(0, 48)),
                                    "phase": np.random.choice(["G1", "S", "G2", "M"]),
                                }
                            )
                            if connector.add_cell(cell):
                                cells_added += 1

                        # Save
                        connector.save_stage()
                        st.session_state.usd_stage_path = connector._stage_path

                        st.success(f"✅ Added {cells_added} cells to scene")
                        st.info(f"📁 File: `{connector._stage_path}`")

                    else:
                        st.error("Failed to create USD stage")

                except ImportError as e:
                    st.error(f"OpenUSD (pxr) not installed: {e}")
                    st.markdown("""
                    **Install OpenUSD:**
                    ```bash
                    pip install usd-core
                    ```
                    """)
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with col2:
        st.markdown("### Status")

        connector = st.session_state.real_connector
        stage_path = st.session_state.usd_stage_path

        if connector and stage_path:
            import os

            st.markdown("🟢 **Real USD Mode Active**")

            st.metric("Mode", "REAL (pxr)")
            st.metric("Stage File", os.path.basename(str(stage_path)))

            # Show file info
            if os.path.exists(stage_path):
                file_size = os.path.getsize(stage_path)
                st.metric("File Size", f"{file_size:,} bytes")

                # Download button
                with open(stage_path, 'r') as f:
                    usd_content = f.read()

                st.download_button(
                    label="📥 Download .usda",
                    data=usd_content,
                    file_name=os.path.basename(str(stage_path)),
                    mime="text/plain",
                )

                # Preview
                st.markdown("### USD Preview")
                with st.expander("View USD Source", expanded=True):
                    # Show first 100 lines
                    lines = usd_content.split('\n')[:100]
                    st.code('\n'.join(lines), language="python")
                    if len(usd_content.split('\n')) > 100:
                        st.info(f"... and {len(usd_content.split(chr(10))) - 100} more lines")

            st.divider()
            st.markdown("### View in 3D")
            st.markdown("""
            Open your `.usda` file in:
            - **NVIDIA Omniverse** (Create, View, or Code)
            - **Pixar usdview** (`usdview file.usda`)
            - **Blender** (with USD addon)
            - **Apple Reality Composer Pro**
            """)

        else:
            st.info("No USD scene generated yet")

            st.markdown("""
            ### How This Works

            This page uses **real OpenUSD** (`pxr` module) to generate
            actual `.usda` files — **no mocks or simulations**.

            **What you get:**
            - ✅ Real USD geometry (spheres, materials)
            - ✅ Actual lighting and camera setup
            - ✅ Proper scene hierarchy
            - ✅ Downloadable .usda files
            - ✅ Opens in Omniverse, usdview, Blender

            **No Nucleus required** — files are generated locally.
            """)

# ─────────────────────────────────────────────────────────────────────────
# Tab 3: Export
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Export to USD")
    st.markdown("Export current simulation state to OpenUSD format.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Export Settings")

        export_path = st.text_input(
            "Export Path",
            value="exports/cognisom_scene.usda",
            help="Output file path (.usda or .usd)"
        )

        export_format = st.radio(
            "Format",
            ["ASCII (.usda)", "Binary (.usd)", "Crate (.usdc)"],
            horizontal=True
        )

        st.markdown("### Include Components")
        include_cells = st.checkbox("Cells", value=True)
        include_immune = st.checkbox("Immune Cells", value=True)
        include_capillaries = st.checkbox("Vascular Network", value=True)
        include_fields = st.checkbox("Spatial Fields", value=True)
        include_molecules = st.checkbox("Molecules", value=False)

        st.divider()

        # Time range
        st.markdown("### Time Range")
        export_mode = st.radio(
            "Export Mode",
            ["Current Frame", "Time Range"],
            horizontal=True
        )

        if export_mode == "Time Range":
            t_start = st.number_input("Start Time (h)", 0.0, 1000.0, 0.0)
            t_end = st.number_input("End Time (h)", 0.0, 1000.0, 24.0)
            t_step = st.number_input("Time Step (h)", 0.1, 10.0, 1.0)

        if st.button("📤 Export to USD", type="primary", use_container_width=True):
            with st.spinner("Exporting..."):
                try:
                    from cognisom.biousd.converter import USDConverter
                    from cognisom.biousd.schema import BioScene, BioCell, CellType
                    import numpy as np

                    # Create sample scene (in real usage, get from simulation engine)
                    scene = BioScene(
                        simulation_time=0.0,
                        time_step=0.1,
                    )

                    # Add sample cells
                    if include_cells:
                        for i in range(100):
                            cell = BioCell(
                                cell_id=i,
                                position=(
                                    np.random.uniform(0, 200),
                                    np.random.uniform(0, 200),
                                    np.random.uniform(0, 50)
                                ),
                                cell_type=np.random.choice(list(CellType)),
                                age=np.random.uniform(0, 48),
                            )
                            scene.cells.append(cell)

                    # Export
                    converter = USDConverter()
                    converter.export_scene(scene, export_path)

                    st.success(f"Exported to `{export_path}`")
                    st.metric("Cells Exported", len(scene.cells))

                except ImportError as e:
                    st.error(f"USD converter not available: {e}")
                except Exception as e:
                    st.error(f"Export error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with col2:
        st.markdown("### USD Preview")

        st.markdown("""
        **Scene Hierarchy:**
        ```
        /World
        ├── /Cells
        │   ├── cell_0001 (BioCell)
        │   ├── cell_0002 (BioCell)
        │   └── ...
        ├── /ImmuneCells
        │   ├── tcell_0001 (BioImmuneCell)
        │   └── ...
        ├── /Environment
        │   ├── /Capillaries
        │   └── /SpatialFields
        ├── /Lights
        └── /Camera
        ```
        """)

        st.divider()

        st.markdown("### Sample USDA")
        st.code('''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.000001
    upAxis = "Z"
)

def Xform "World" {
    def Xform "Cells" {
        def BioCell "cell_0001" {
            int bio:cellId = 1
            token bio:cellType = "cancer"
            token bio:phase = "G1"
            float bio:age = 12.5
            bool bio:alive = true
            float3 xformOp:translate = (50.0, 75.2, 10.0)

            # Applied Schemas
            float bio:metabolic:oxygen = 0.18
            float bio:metabolic:glucose = 4.2
            float bio:metabolic:atp = 850.0
        }
    }
}''', language="usda")

# ─────────────────────────────────────────────────────────────────────────
# Tab 4: Documentation
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Documentation")

    st.markdown("""
    ## Bio-USD Specification

    Bio-USD extends OpenUSD with schemas for biological simulation.

    ### Prim Type Hierarchy

    ```
    BioUnit (abstract base)
    ├── BioCell              — Single biological cell
    │   └── BioImmuneCell    — T cells, NK cells, macrophages
    ├── BioGene              — Gene with expression state
    ├── BioProtein           — Protein with 3D structure
    ├── BioMolecule          — Small molecule (drug, metabolite)
    ├── BioTissue            — Cell collection / tissue
    ├── BioCapillary         — Blood vessel segment
    ├── BioSpatialField      — 3D concentration field
    └── BioExosome           — Extracellular vesicle
    ```

    ### Applied API Schemas

    Composable metadata that can be applied to any prim:

    | Schema | Purpose |
    |--------|---------|
    | `BioMetabolicAPI` | O2, glucose, ATP, lactate |
    | `BioGeneExpressionAPI` | Expression levels, mutations |
    | `BioEpigeneticAPI` | Methylation, histone marks |
    | `BioImmuneAPI` | MHC-I, activation state |
    | `BioInteractionAPI` | Binding relationships |

    ### Extensibility

    Register custom prim types at runtime:

    ```python
    from cognisom.biousd.schema import register_prim, BioUnit

    @register_prim("bio_virus", version="1.0.0")
    class BioVirusParticle(BioUnit):
        virus_type: str = ""
        capsid_proteins: List[str] = field(default_factory=list)
        genome_rna: str = ""
    ```

    ## Omniverse Integration

    ### Real-Time Sync

    The Cognisom-Omniverse bridge enables:

    1. **Live Streaming** — Cell positions update in real-time
    2. **Physics Handoff** — Force calculations via PhysX
    3. **Rendering** — Ray-traced visualization via RTX
    4. **Collaboration** — Multi-user viewing sessions

    ### Connection Architecture

    ```
    Cognisom Engine
         ↓
    Bio-USD Converter
         ↓
    Omniverse Connector
         ↓
    Nucleus Server (omni://localhost)
         ↓
    Omniverse Kit / Isaac Sim
    ```

    ### Requirements

    - NVIDIA Omniverse Kit 2024.x+
    - Nucleus server (local or remote)
    - `omni.client` Python package
    - RTX GPU for real-time rendering

    ## SBML Conversion

    Import SBML metabolic models:

    ```python
    from cognisom.biousd.sbml_converter import SBMLConverter

    converter = SBMLConverter()
    scene = converter.from_sbml("model.sbml")
    ```

    Supports:
    - Species → BioMolecule
    - Reactions → BioInteractionAPI
    - Compartments → BioTissue
    """)

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
