"""
UI Panel for Cognisom Simulation Extension
==========================================

Provides the user interface for controlling the simulation.
"""

import omni.ui as ui
from typing import Optional

from .simulation_manager import SimulationManager


class CognisomPanel:
    """UI panel for Cognisom simulation control."""

    def __init__(self, simulation_manager: SimulationManager):
        self._sim = simulation_manager
        self._widgets = {}
        self._build_ui()

    def _build_ui(self):
        """Build the UI layout."""
        with ui.VStack(spacing=10):
            # Header
            self._build_header()

            # Control buttons
            self._build_controls()

            # Parameters section
            self._build_parameters()

            # Statistics section
            self._build_statistics()

            # Advanced options
            self._build_advanced()

    def _build_header(self):
        """Build header section."""
        with ui.HStack(height=40):
            ui.Spacer(width=10)
            ui.Image(
                "",  # Would be icon path
                width=32,
                height=32
            )
            ui.Spacer(width=10)
            ui.Label(
                "Cognisom Biological Simulation",
                style={"font_size": 18, "color": 0xFF00A0FF}
            )
            ui.Spacer()

    def _build_controls(self):
        """Build simulation control buttons."""
        with ui.CollapsableFrame("Simulation Control", collapsed=False):
            with ui.VStack(spacing=5):
                with ui.HStack(spacing=5, height=30):
                    # Start/Stop button
                    self._widgets["start_btn"] = ui.Button(
                        "Start",
                        clicked_fn=self._on_start_clicked,
                        style={"background_color": 0xFF2A7F2A}
                    )

                    # Pause/Resume button
                    self._widgets["pause_btn"] = ui.Button(
                        "Pause",
                        clicked_fn=self._on_pause_clicked,
                        enabled=False
                    )

                    # Reset button
                    self._widgets["reset_btn"] = ui.Button(
                        "Reset",
                        clicked_fn=self._on_reset_clicked
                    )

                # Status indicator
                with ui.HStack(height=20):
                    ui.Label("Status:", width=60)
                    self._widgets["status_label"] = ui.Label(
                        "Stopped",
                        style={"color": 0xFFAAAAAA}
                    )

    def _build_parameters(self):
        """Build parameters section."""
        with ui.CollapsableFrame("Simulation Parameters", collapsed=False):
            with ui.VStack(spacing=8):
                # Cell count
                self._build_slider(
                    "Cell Count",
                    "cell_count",
                    min_val=10,
                    max_val=1000,
                    default=100,
                    step=10
                )

                # Division rate
                self._build_slider(
                    "Division Rate",
                    "division_rate",
                    min_val=0.0,
                    max_val=1.0,
                    default=0.1,
                    step=0.01
                )

                # Death rate
                self._build_slider(
                    "Death Rate",
                    "death_rate",
                    min_val=0.0,
                    max_val=1.0,
                    default=0.05,
                    step=0.01
                )

                # Migration speed
                self._build_slider(
                    "Migration Speed",
                    "migration_speed",
                    min_val=0.0,
                    max_val=10.0,
                    default=2.0,
                    step=0.1
                )

                # Interaction radius
                self._build_slider(
                    "Interaction Radius",
                    "interaction_radius",
                    min_val=5.0,
                    max_val=100.0,
                    default=20.0,
                    step=1.0
                )

    def _build_slider(
        self,
        label: str,
        param_name: str,
        min_val: float,
        max_val: float,
        default: float,
        step: float
    ):
        """Build a labeled slider."""
        with ui.HStack(height=24):
            ui.Label(label, width=120)

            slider = ui.FloatSlider(
                min=min_val,
                max=max_val,
                step=step
            )
            slider.model.set_value(default)

            value_label = ui.Label(f"{default:.2f}", width=50)

            # Bind slider to parameter
            def on_value_changed(model, param=param_name, lbl=value_label):
                value = model.get_value_as_float()
                lbl.text = f"{value:.2f}"
                if self._sim:
                    self._sim.set_param(param, value)

            slider.model.add_value_changed_fn(on_value_changed)

            self._widgets[f"slider_{param_name}"] = slider

    def _build_statistics(self):
        """Build statistics display."""
        with ui.CollapsableFrame("Statistics", collapsed=False):
            with ui.VStack(spacing=5):
                # Create stat rows
                self._widgets["stat_cells"] = self._build_stat_row("Total Cells", "0")
                self._widgets["stat_dividing"] = self._build_stat_row("Dividing", "0")
                self._widgets["stat_apoptotic"] = self._build_stat_row("Apoptotic", "0")
                self._widgets["stat_fps"] = self._build_stat_row("FPS", "0.0")
                self._widgets["stat_step_time"] = self._build_stat_row("Step Time", "0.0 ms")

    def _build_stat_row(self, label: str, initial: str) -> ui.Label:
        """Build a statistics row."""
        with ui.HStack(height=20):
            ui.Label(f"{label}:", width=100)
            value_label = ui.Label(initial, style={"color": 0xFF00FF00})
            ui.Spacer()
        return value_label

    def _build_advanced(self):
        """Build advanced options section."""
        with ui.CollapsableFrame("Advanced", collapsed=True):
            with ui.VStack(spacing=5):
                # Physics toggle
                with ui.HStack(height=24):
                    ui.Label("Enable Physics", width=120)
                    self._widgets["physics_checkbox"] = ui.CheckBox()
                    self._widgets["physics_checkbox"].model.set_value(True)

                # Trails toggle
                with ui.HStack(height=24):
                    ui.Label("Show Trails", width=120)
                    self._widgets["trails_checkbox"] = ui.CheckBox()
                    self._widgets["trails_checkbox"].model.set_value(False)

                # Export button
                ui.Button(
                    "Export State",
                    clicked_fn=self._on_export_clicked,
                    height=30
                )

                # Connect to backend
                ui.Button(
                    "Connect to Cognisom Backend",
                    clicked_fn=self._on_connect_clicked,
                    height=30
                )

    # ── Event Handlers ──────────────────────────────────────────────────

    def _on_start_clicked(self):
        """Handle start button click."""
        if not self._sim:
            return

        if self._sim.is_running:
            self._sim.stop()
            self._widgets["start_btn"].text = "Start"
            self._widgets["start_btn"].style = {"background_color": 0xFF2A7F2A}
            self._widgets["pause_btn"].enabled = False
            self._widgets["status_label"].text = "Stopped"
            self._widgets["status_label"].style = {"color": 0xFFAAAAAA}
        else:
            self._sim.start()
            self._widgets["start_btn"].text = "Stop"
            self._widgets["start_btn"].style = {"background_color": 0xFF7F2A2A}
            self._widgets["pause_btn"].enabled = True
            self._widgets["status_label"].text = "Running"
            self._widgets["status_label"].style = {"color": 0xFF00FF00}

    def _on_pause_clicked(self):
        """Handle pause button click."""
        if not self._sim:
            return

        if self._sim.is_paused:
            self._sim.resume()
            self._widgets["pause_btn"].text = "Pause"
            self._widgets["status_label"].text = "Running"
            self._widgets["status_label"].style = {"color": 0xFF00FF00}
        else:
            self._sim.pause()
            self._widgets["pause_btn"].text = "Resume"
            self._widgets["status_label"].text = "Paused"
            self._widgets["status_label"].style = {"color": 0xFFFFFF00}

    def _on_reset_clicked(self):
        """Handle reset button click."""
        if self._sim:
            self._sim.reset()

    def _on_export_clicked(self):
        """Handle export button click."""
        # Would open file dialog and export state
        pass

    def _on_connect_clicked(self):
        """Handle connect button click."""
        # Would connect to Cognisom backend
        pass

    # ── Updates ─────────────────────────────────────────────────────────

    def update_stats(self):
        """Update statistics display."""
        if not self._sim:
            return

        stats = self._sim.stats

        if "stat_cells" in self._widgets:
            self._widgets["stat_cells"].text = str(stats.get("total_cells", 0))

        if "stat_dividing" in self._widgets:
            self._widgets["stat_dividing"].text = str(stats.get("dividing", 0))

        if "stat_apoptotic" in self._widgets:
            self._widgets["stat_apoptotic"].text = str(stats.get("apoptotic", 0))

        if "stat_fps" in self._widgets:
            self._widgets["stat_fps"].text = f"{stats.get('fps', 0):.1f}"

        if "stat_step_time" in self._widgets:
            self._widgets["stat_step_time"].text = f"{stats.get('step_time_ms', 0):.2f} ms"
