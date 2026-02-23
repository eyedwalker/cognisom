"""
Cognisom Simulation Extension
=============================

Main extension class for Omniverse Kit integration.
Supports two modes:
  - Cell Simulation: Generic tumor microenvironment (SimulationManager)
  - Diapedesis: Leukocyte extravasation cascade playback (DiapedesisManager)
"""

import carb
import omni.ext
import omni.ui as ui
import omni.usd

from .simulation_manager import SimulationManager
from .diapedesis_manager import DiapedesisManager
from .streaming_server import StreamingServer
from .ui_panel import CognisomPanel


class CognisomSimExtension(omni.ext.IExt):
    """Cognisom biological simulation extension for Omniverse."""

    def __init__(self):
        super().__init__()
        self._window = None
        self._panel = None
        self._simulation_manager = None
        self._diapedesis_manager = None
        self._streaming_server = None
        self._menu_items = []
        self._update_sub = None
        self._mode = "cell_sim"  # "cell_sim" or "diapedesis"

    def on_startup(self, ext_id: str):
        """Called when extension is loaded."""
        carb.log_info("[cognisom.sim] Starting Cognisom Simulation Extension")

        # Get extension settings
        settings = carb.settings.get_settings()
        self._fps = settings.get_as_float("exts/cognisom.sim/simulation/fps") or 60.0
        self._dt = settings.get_as_float("exts/cognisom.sim/simulation/dt") or 0.01

        # Initialize both managers
        self._simulation_manager = SimulationManager()
        self._diapedesis_manager = DiapedesisManager()

        # Start HTTP streaming server (port 8211)
        self._streaming_server = StreamingServer(
            self._diapedesis_manager, port=8211)
        self._streaming_server.start()

        # Create UI window
        self._create_ui()

        # Add menu items
        self._add_menu()

        # Subscribe to updates
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="cognisom.sim.update"
            )
        )

        carb.log_info("[cognisom.sim] Extension started successfully")

    def on_shutdown(self):
        """Called when extension is unloaded."""
        carb.log_info("[cognisom.sim] Shutting down Cognisom Simulation Extension")

        # Stop simulations
        if self._simulation_manager:
            self._simulation_manager.stop()
            self._simulation_manager = None

        if self._streaming_server:
            self._streaming_server.shutdown()
            self._streaming_server = None

        if self._diapedesis_manager:
            self._diapedesis_manager.stop()
            self._diapedesis_manager.clear()
            self._diapedesis_manager = None

        # Remove update subscription
        if self._update_sub:
            self._update_sub = None

        # Clean up UI
        if self._window:
            self._window.destroy()
            self._window = None

        # Remove menu items
        for item in self._menu_items:
            item.destroy()
        self._menu_items.clear()

        carb.log_info("[cognisom.sim] Extension shutdown complete")

    def _create_ui(self):
        """Create the extension UI window."""
        self._window = ui.Window(
            "Cognisom Simulation",
            width=420,
            height=700,
            visible=True,
            dockPreference=ui.DockPreference.RIGHT_BOTTOM
        )

        with self._window.frame:
            self._panel = CognisomPanel(
                self._simulation_manager,
                self._diapedesis_manager,
                mode_changed_fn=self._on_mode_changed,
            )

    def _add_menu(self):
        """Add extension to Omniverse menu."""
        editor_menu = omni.kit.ui.get_editor_menu()

        menu_item = editor_menu.add_item(
            "Window/Simulation/Cognisom",
            self._on_menu_click,
            toggle=True,
            value=True
        )
        self._menu_items.append(menu_item)

        # Diapedesis shortcut
        diap_item = editor_menu.add_item(
            "Window/Simulation/Cognisom Diapedesis",
            self._on_diapedesis_menu_click,
        )
        self._menu_items.append(diap_item)

    def _on_menu_click(self, menu_item, value):
        """Handle menu click."""
        if self._window:
            self._window.visible = value

    def _on_diapedesis_menu_click(self, menu_item, value):
        """Switch to diapedesis mode and show window."""
        self._mode = "diapedesis"
        if self._panel:
            self._panel.set_mode("diapedesis")
        if self._window:
            self._window.visible = True

    def _on_mode_changed(self, mode: str):
        """Called when UI switches between cell_sim and diapedesis."""
        self._mode = mode
        carb.log_info(f"[cognisom.sim] Mode changed to: {mode}")

    def _on_update(self, event):
        """Called every frame."""
        dt = event.payload.get("dt", 1.0 / 60.0)

        if self._mode == "cell_sim":
            if self._simulation_manager and self._simulation_manager.is_running:
                self._simulation_manager.update(dt)
        elif self._mode == "diapedesis":
            if self._diapedesis_manager and self._diapedesis_manager.is_playing:
                self._diapedesis_manager.update(dt)

        # Update UI
        if self._panel:
            self._panel.update_stats()


# Extension entry point
def get_extension():
    """Return extension instance."""
    return CognisomSimExtension()
