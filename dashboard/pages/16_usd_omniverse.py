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
# Tab 2: Omniverse Connection
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Omniverse Connection")
    st.markdown("Connect to NVIDIA Omniverse for real-time 3D visualization.")

    # Initialize session state
    if "omni_connector" not in st.session_state:
        st.session_state.omni_connector = None

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Connection Settings")

        import os

        # Auto-detect AWS deployment
        is_aws = os.environ.get("OMNIVERSE_URL") is not None
        aws_url = os.environ.get("OMNIVERSE_URL", "omniverse://localhost:3019/cognisom")

        # Deployment mode presets
        deploy_options = ["AWS GPU (Auto)", "Local Docker", "Local Omniverse", "Omniverse Cloud"] if is_aws else \
                         ["Local Docker", "Local Omniverse", "Omniverse Cloud"]

        deploy_mode = st.radio(
            "Deployment Mode",
            deploy_options,
            horizontal=True,
            help="Select where Nucleus is running"
        )

        url_presets = {
            "AWS GPU (Auto)": aws_url,
            "Local Docker": "omniverse://nucleus:3019/cognisom",
            "Local Omniverse": "omniverse://localhost/cognisom",
            "Omniverse Cloud": "omniverse://cloud.nvidia.com/your-org/cognisom",
        }

        omni_url = st.text_input(
            "Omniverse URL",
            value=url_presets.get(deploy_mode, aws_url if is_aws else "omniverse://localhost/cognisom"),
            help="URL to Omniverse Nucleus server"
        )

        stage_name = st.text_input(
            "Stage Name",
            value="cognisom_simulation.usd",
            help="USD stage file name"
        )

        # Show deployment-specific instructions
        if is_aws and deploy_mode == "AWS GPU (Auto)":
            st.success("Running on AWS with GPU — Nucleus is ready!")
        elif deploy_mode == "Local Docker":
            with st.expander("Docker Setup Instructions", expanded=True):
                st.markdown("""
                **Start Nucleus container:**
                ```bash
                # From project root:
                docker-compose --profile omniverse up -d nucleus

                # Check status:
                docker logs cognisom-nucleus
                ```

                **Nucleus Web UI:** http://localhost:3009
                """)

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("🔌 Connect", type="primary", use_container_width=True):
                try:
                    from cognisom.omniverse.connector import OmniverseConnector, ConnectionConfig

                    config = ConnectionConfig(
                        url=omni_url,
                        stage_name=stage_name,
                    )

                    connector = OmniverseConnector(config)
                    success = connector.connect()

                    if success:
                        st.session_state.omni_connector = connector
                        st.success("Connected to Omniverse!")
                    else:
                        st.error("Connection failed")

                except ImportError as e:
                    st.error(f"Omniverse module not available: {e}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        with col_btn2:
            if st.button("🔌 Disconnect", use_container_width=True):
                if st.session_state.omni_connector:
                    st.session_state.omni_connector.disconnect()
                    st.session_state.omni_connector = None
                    st.info("Disconnected")

        st.divider()

        # Advanced settings
        with st.expander("Advanced Settings"):
            auto_reconnect = st.checkbox("Auto-reconnect", value=True)
            heartbeat = st.slider("Heartbeat Interval (s)", 5, 60, 10)
            timeout = st.slider("Connection Timeout (s)", 10, 120, 30)

    with col2:
        st.markdown("### Connection Status")

        connector = st.session_state.omni_connector

        if connector:
            info = connector.get_info()

            # Status indicator
            status = info.get("status", "unknown")
            status_colors = {
                "connected": "🟢",
                "authenticated": "🟢",
                "disconnected": "🔴",
                "connecting": "🟡",
                "reconnecting": "🟡",
                "error": "🔴",
            }

            st.markdown(f"**Status:** {status_colors.get(status, '⚪')} {status.title()}")

            # Connection info
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Connection Attempts", info.get("connection_attempts", 0))
            with col_s2:
                simulated = "Yes" if info.get("simulated") else "No"
                st.metric("Simulated Mode", simulated)

            st.markdown(f"**URL:** `{info.get('url', 'N/A')}`")
            st.markdown(f"**Stage:** `{info.get('stage_name', 'N/A')}`")
            st.markdown(f"**Has Stage:** {'Yes' if info.get('has_stage') else 'No'}")

            # Event history
            st.markdown("### Recent Events")
            events = connector.get_event_history(limit=10)
            if events:
                for event in reversed(events):
                    st.text(f"[{event.event_type}] {event.message}")
            else:
                st.info("No events yet")

        else:
            st.info("Not connected to Omniverse")

            st.markdown("""
            **Requirements:**
            - NVIDIA Omniverse Kit installed
            - Nucleus server running
            - `omni.client` Python package

            **Without Omniverse:**
            - Runs in simulation mode
            - Mock USD stage for testing
            - No real-time visualization
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
st.divider()
st.caption("Cognisom USD & Omniverse | OpenUSD + NVIDIA Omniverse | eyentelligence inc.")
