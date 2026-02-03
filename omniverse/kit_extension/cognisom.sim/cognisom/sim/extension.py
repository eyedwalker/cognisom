"""
Cognisom Simulation Extension
=============================

Main extension class for Omniverse Kit integration.
"""

import asyncio
import carb
import omni.ext
import omni.ui as ui
import omni.usd
from pxr import Usd, UsdGeom, Gf

from .simulation_manager import SimulationManager
from .ui_panel import CognisomPanel


class CognisomSimExtension(omni.ext.IExt):
    """Cognisom biological simulation extension for Omniverse."""

    def __init__(self):
        super().__init__()
        self._window = None
        self._panel = None
        self._simulation_manager = None
        self._menu_items = []
        self._update_sub = None

    def on_startup(self, ext_id: str):
        """Called when extension is loaded."""
        carb.log_info("[cognisom.sim] Starting Cognisom Simulation Extension")

        # Get extension settings
        settings = carb.settings.get_settings()
        self._fps = settings.get_as_float("exts/cognisom.sim/simulation/fps") or 60.0
        self._dt = settings.get_as_float("exts/cognisom.sim/simulation/dt") or 0.01

        # Initialize simulation manager
        self._simulation_manager = SimulationManager()

        # Create UI window
        self._create_ui()

        # Add menu item
        self._add_menu()

        # Subscribe to updates
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="cognisom.sim.update"
        )

        carb.log_info("[cognisom.sim] Extension started successfully")

    def on_shutdown(self):
        """Called when extension is unloaded."""
        carb.log_info("[cognisom.sim] Shutting down Cognisom Simulation Extension")

        # Stop simulation
        if self._simulation_manager:
            self._simulation_manager.stop()
            self._simulation_manager = None

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
            width=400,
            height=600,
            visible=True,
            dockPreference=ui.DockPreference.RIGHT_BOTTOM
        )

        with self._window.frame:
            self._panel = CognisomPanel(self._simulation_manager)

    def _add_menu(self):
        """Add extension to Omniverse menu."""
        editor_menu = omni.kit.ui.get_editor_menu()

        # Main menu item
        menu_item = editor_menu.add_item(
            "Window/Simulation/Cognisom",
            self._on_menu_click,
            toggle=True,
            value=True
        )
        self._menu_items.append(menu_item)

    def _on_menu_click(self, menu_item, value):
        """Handle menu click."""
        if self._window:
            self._window.visible = value

    def _on_update(self, event):
        """Called every frame."""
        if self._simulation_manager and self._simulation_manager.is_running:
            # Update simulation
            dt = event.payload.get("dt", 1.0 / 60.0)
            self._simulation_manager.update(dt)

            # Update UI
            if self._panel:
                self._panel.update_stats()


# Extension entry point
def get_extension():
    """Return extension instance."""
    return CognisomSimExtension()
