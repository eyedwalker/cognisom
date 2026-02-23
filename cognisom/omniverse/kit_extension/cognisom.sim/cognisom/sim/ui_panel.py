"""
UI Panel for Cognisom Simulation Extension
==========================================

Provides the user interface for controlling both cell simulation
and diapedesis playback modes.
"""

import omni.ui as ui
from typing import Callable, Optional

from .simulation_manager import SimulationManager


class CognisomPanel:
    """UI panel for Cognisom simulation control."""

    def __init__(
        self,
        simulation_manager: "SimulationManager",
        diapedesis_manager=None,
        mode_changed_fn: Optional[Callable] = None,
    ):
        self._sim = simulation_manager
        self._diap = diapedesis_manager
        self._mode_changed_fn = mode_changed_fn
        self._mode = "cell_sim"
        self._widgets = {}
        self._build_ui()

    def set_mode(self, mode: str):
        """Switch between cell_sim and diapedesis."""
        self._mode = mode
        self._update_mode_visibility()

    def _build_ui(self):
        """Build the UI layout."""
        with ui.VStack(spacing=8):
            # Header
            self._build_header()

            # Mode selector
            self._build_mode_selector()

            # Cell sim section (shown when mode = cell_sim)
            self._cell_sim_frame = ui.CollapsableFrame(
                "Cell Simulation", collapsed=False, visible=True)
            with self._cell_sim_frame:
                with ui.VStack(spacing=5):
                    self._build_cell_sim_controls()
                    self._build_cell_sim_params()
                    self._build_cell_sim_stats()

            # Diapedesis section (shown when mode = diapedesis)
            self._diap_frame = ui.CollapsableFrame(
                "Diapedesis Cascade", collapsed=False, visible=False)
            with self._diap_frame:
                with ui.VStack(spacing=5):
                    self._build_diap_presets()
                    self._build_diap_controls()
                    self._build_diap_timeline()
                    self._build_diap_cascade_steps()
                    self._build_diap_stats()

            # Advanced options (shared)
            self._build_advanced()

    def _build_header(self):
        """Build header section."""
        with ui.HStack(height=40):
            ui.Spacer(width=10)
            ui.Label(
                "Cognisom Biological Simulation",
                style={"font_size": 18, "color": 0xFF00A0FF}
            )
            ui.Spacer()

    # ── Mode Selector ──────────────────────────────────────────────────

    def _build_mode_selector(self):
        """Build mode toggle between cell sim and diapedesis."""
        with ui.HStack(height=30, spacing=5):
            ui.Label("Mode:", width=50)

            self._widgets["mode_cell"] = ui.Button(
                "Cell Sim",
                clicked_fn=lambda: self._switch_mode("cell_sim"),
                style={"background_color": 0xFF2A7F2A},
                width=120,
            )
            self._widgets["mode_diap"] = ui.Button(
                "Diapedesis",
                clicked_fn=lambda: self._switch_mode("diapedesis"),
                style={"background_color": 0xFF555555},
                width=120,
            )

    def _switch_mode(self, mode: str):
        """Handle mode switch."""
        self._mode = mode
        self._update_mode_visibility()
        if self._mode_changed_fn:
            self._mode_changed_fn(mode)

    def _update_mode_visibility(self):
        """Show/hide sections based on mode."""
        is_cell = self._mode == "cell_sim"
        self._cell_sim_frame.visible = is_cell
        self._diap_frame.visible = not is_cell

        # Update button styles
        active = 0xFF2A7F2A
        inactive = 0xFF555555
        if "mode_cell" in self._widgets:
            self._widgets["mode_cell"].style = {
                "background_color": active if is_cell else inactive}
        if "mode_diap" in self._widgets:
            self._widgets["mode_diap"].style = {
                "background_color": inactive if is_cell else active}

    # ── Cell Simulation Controls ───────────────────────────────────────

    def _build_cell_sim_controls(self):
        """Build cell simulation control buttons."""
        with ui.HStack(spacing=5, height=30):
            self._widgets["start_btn"] = ui.Button(
                "Start",
                clicked_fn=self._on_start_clicked,
                style={"background_color": 0xFF2A7F2A}
            )
            self._widgets["pause_btn"] = ui.Button(
                "Pause",
                clicked_fn=self._on_pause_clicked,
                enabled=False
            )
            self._widgets["reset_btn"] = ui.Button(
                "Reset",
                clicked_fn=self._on_reset_clicked
            )

        with ui.HStack(height=20):
            ui.Label("Status:", width=60)
            self._widgets["status_label"] = ui.Label(
                "Stopped",
                style={"color": 0xFFAAAAAA}
            )

    def _build_cell_sim_params(self):
        """Build cell simulation parameters."""
        with ui.CollapsableFrame("Parameters", collapsed=True):
            with ui.VStack(spacing=8):
                self._build_slider("Cell Count", "cell_count", 10, 1000, 100, 10)
                self._build_slider("Division Rate", "division_rate", 0.0, 1.0, 0.1, 0.01)
                self._build_slider("Death Rate", "death_rate", 0.0, 1.0, 0.05, 0.01)
                self._build_slider("Migration Speed", "migration_speed", 0.0, 10.0, 2.0, 0.1)

    def _build_cell_sim_stats(self):
        """Build cell simulation statistics."""
        with ui.CollapsableFrame("Statistics", collapsed=False):
            with ui.VStack(spacing=5):
                self._widgets["stat_cells"] = self._build_stat_row("Total Cells", "0")
                self._widgets["stat_dividing"] = self._build_stat_row("Dividing", "0")
                self._widgets["stat_apoptotic"] = self._build_stat_row("Apoptotic", "0")
                self._widgets["stat_fps"] = self._build_stat_row("FPS", "0.0")
                self._widgets["stat_step_time"] = self._build_stat_row("Step Time", "0.0 ms")

    # ── Diapedesis Controls ──────────────────────────────────────────────

    def _build_diap_presets(self):
        """Build preset selection for diapedesis."""
        with ui.CollapsableFrame("Preset Scenarios", collapsed=False):
            with ui.VStack(spacing=4):
                presets = [
                    ("healthy", "Healthy Vessel"),
                    ("inflammation", "Acute Inflammation"),
                    ("lad1", "LAD-1 (Selectin Deficiency)"),
                    ("lad2", "LAD-2 (Fucose Deficiency)"),
                    ("lad3", "LAD-3 (Integrin Defect)"),
                ]
                for key, label in presets:
                    style = {"background_color": 0xFF2A4F7F}
                    if key == "inflammation":
                        style["background_color"] = 0xFF7F3F1F
                    btn = ui.Button(
                        label,
                        clicked_fn=lambda k=key: self._on_diap_preset(k),
                        height=26,
                        style=style,
                    )
                    self._widgets[f"preset_{key}"] = btn

                # Duration slider
                with ui.HStack(height=24):
                    ui.Label("Duration (s):", width=100)
                    self._widgets["diap_duration"] = ui.FloatSlider(
                        min=30, max=300, step=10)
                    self._widgets["diap_duration"].model.set_value(120)
                    self._widgets["diap_duration_lbl"] = ui.Label("120", width=40)

                    def on_dur(model):
                        self._widgets["diap_duration_lbl"].text = f"{model.get_value_as_float():.0f}"
                    self._widgets["diap_duration"].model.add_value_changed_fn(on_dur)

    def _build_diap_controls(self):
        """Build diapedesis playback controls."""
        with ui.HStack(spacing=5, height=30):
            self._widgets["diap_play"] = ui.Button(
                "Play",
                clicked_fn=self._on_diap_play,
                style={"background_color": 0xFF2A7F2A},
            )
            self._widgets["diap_pause"] = ui.Button(
                "Pause",
                clicked_fn=self._on_diap_pause,
                enabled=False,
            )
            self._widgets["diap_stop"] = ui.Button(
                "Stop",
                clicked_fn=self._on_diap_stop,
            )
            self._widgets["diap_step_back"] = ui.Button(
                "<",
                clicked_fn=self._on_diap_step_back,
                width=30,
            )
            self._widgets["diap_step_fwd"] = ui.Button(
                ">",
                clicked_fn=self._on_diap_step_fwd,
                width=30,
            )

        # Speed control
        with ui.HStack(height=24):
            ui.Label("Speed:", width=50)
            self._widgets["diap_speed"] = ui.FloatSlider(min=0.1, max=5.0, step=0.1)
            self._widgets["diap_speed"].model.set_value(1.0)
            self._widgets["diap_speed_lbl"] = ui.Label("1.0x", width=40)

            def on_speed(model):
                val = model.get_value_as_float()
                self._widgets["diap_speed_lbl"].text = f"{val:.1f}x"
                if self._diap:
                    self._diap.playback_speed = val
            self._widgets["diap_speed"].model.add_value_changed_fn(on_speed)

        # Status
        with ui.HStack(height=20):
            ui.Label("Status:", width=50)
            self._widgets["diap_status"] = ui.Label(
                "No frames loaded",
                style={"color": 0xFFAAAAAA}
            )

    def _build_diap_timeline(self):
        """Build frame scrubber timeline."""
        with ui.CollapsableFrame("Timeline", collapsed=False):
            with ui.VStack(spacing=4):
                self._widgets["diap_timeline"] = ui.FloatSlider(
                    min=0, max=1, step=1)
                self._widgets["diap_timeline"].model.set_value(0)

                def on_seek(model):
                    if self._diap and self._diap.total_frames > 0:
                        frame = int(model.get_value_as_float())
                        self._diap.seek(frame)
                self._widgets["diap_timeline"].model.add_value_changed_fn(on_seek)

                with ui.HStack(height=18):
                    self._widgets["diap_frame_lbl"] = ui.Label(
                        "Frame: 0 / 0", style={"font_size": 12})
                    ui.Spacer()
                    self._widgets["diap_time_lbl"] = ui.Label(
                        "Time: 0.0s", style={"font_size": 12})

    def _build_diap_cascade_steps(self):
        """Build diapedesis cascade step indicators."""
        with ui.CollapsableFrame("Cascade Steps", collapsed=False):
            with ui.VStack(spacing=3):
                steps = [
                    ("1. Cytokine Activation", 0xFF4488FF),
                    ("2. Selectin Rolling", 0xFFFFDD00),
                    ("3. Chemokine Signaling", 0xFF44DDFF),
                    ("4. Integrin Activation", 0xFF00FFEE),
                    ("5. Firm Adhesion / Crawling", 0xFFFF5500),
                    ("6. Transmigration", 0xFFCC00AA),
                    ("7. Phagocytosis", 0xFF44DD44),
                ]
                for label, color in steps:
                    with ui.HStack(height=18):
                        indicator = ui.Rectangle(
                            width=8, height=8,
                            style={"background_color": color, "border_radius": 4})
                        ui.Spacer(width=6)
                        lbl = ui.Label(label, style={"font_size": 12})
                        self._widgets[f"step_{label[:1]}"] = lbl

    def _build_diap_stats(self):
        """Build diapedesis statistics display."""
        with ui.CollapsableFrame("Metrics", collapsed=False):
            with ui.VStack(spacing=5):
                self._widgets["diap_stat_leuko"] = self._build_stat_row(
                    "Leukocytes", "0")
                self._widgets["diap_stat_rbc"] = self._build_stat_row(
                    "RBCs", "0")
                self._widgets["diap_stat_bacteria"] = self._build_stat_row(
                    "Bacteria Alive", "0")
                self._widgets["diap_stat_fps"] = self._build_stat_row(
                    "Render FPS", "0.0")
                self._widgets["diap_stat_frame_ms"] = self._build_stat_row(
                    "Frame Time", "0.0 ms")

    # ── Shared Widgets ─────────────────────────────────────────────────

    def _build_slider(self, label, param_name, min_val, max_val, default, step):
        """Build a labeled slider."""
        with ui.HStack(height=24):
            ui.Label(label, width=120)
            slider = ui.FloatSlider(min=min_val, max=max_val, step=step)
            slider.model.set_value(default)
            value_label = ui.Label(f"{default:.2f}", width=50)

            def on_value_changed(model, param=param_name, lbl=value_label):
                value = model.get_value_as_float()
                lbl.text = f"{value:.2f}"
                if self._sim:
                    self._sim.set_param(param, value)
            slider.model.add_value_changed_fn(on_value_changed)
            self._widgets[f"slider_{param_name}"] = slider

    def _build_stat_row(self, label: str, initial: str) -> ui.Label:
        """Build a statistics row."""
        with ui.HStack(height=20):
            ui.Label(f"{label}:", width=120)
            value_label = ui.Label(initial, style={"color": 0xFF00FF00})
            ui.Spacer()
        return value_label

    def _build_advanced(self):
        """Build advanced options section."""
        with ui.CollapsableFrame("Advanced", collapsed=True):
            with ui.VStack(spacing=5):
                with ui.HStack(height=24):
                    ui.Label("Enable Physics", width=120)
                    self._widgets["physics_checkbox"] = ui.CheckBox()
                    self._widgets["physics_checkbox"].model.set_value(True)

                with ui.HStack(height=24):
                    ui.Label("Show Trails", width=120)
                    self._widgets["trails_checkbox"] = ui.CheckBox()
                    self._widgets["trails_checkbox"].model.set_value(False)

                ui.Button(
                    "Export USD Scene",
                    clicked_fn=self._on_export_clicked,
                    height=30
                )
                ui.Button(
                    "Connect to Cognisom Backend",
                    clicked_fn=self._on_connect_clicked,
                    height=30
                )

    # ── Cell Sim Event Handlers ──────────────────────────────────────────

    def _on_start_clicked(self):
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
        if self._sim:
            self._sim.reset()

    # ── Diapedesis Event Handlers ──────────────────────────────────────

    def _on_diap_preset(self, preset_name: str):
        """Load a diapedesis preset and build the scene."""
        if not self._diap:
            return

        # Stop any running playback
        self._diap.stop()

        duration = 120.0
        if "diap_duration" in self._widgets:
            duration = self._widgets["diap_duration"].model.get_value_as_float()

        self._widgets["diap_status"].text = f"Computing {preset_name}..."
        self._widgets["diap_status"].style = {"color": 0xFFFFFF00}

        success = self._diap.load_preset(preset_name, duration=duration)
        if success:
            self._diap.build_scene()
            total = self._diap.total_frames
            self._widgets["diap_status"].text = f"Ready: {total} frames ({preset_name})"
            self._widgets["diap_status"].style = {"color": 0xFF00FF00}
            # Update timeline range
            self._widgets["diap_timeline"].max = max(1, total - 1)
            self._widgets["diap_timeline"].model.set_value(0)
            self._widgets["diap_frame_lbl"].text = f"Frame: 0 / {total}"
        else:
            self._widgets["diap_status"].text = "Failed to load preset"
            self._widgets["diap_status"].style = {"color": 0xFFFF0000}

    def _on_diap_play(self):
        if not self._diap:
            return
        if self._diap.is_paused:
            self._diap.resume()
        else:
            self._diap.play()
        self._widgets["diap_play"].text = "Playing"
        self._widgets["diap_play"].style = {"background_color": 0xFF7F2A2A}
        self._widgets["diap_pause"].enabled = True
        self._widgets["diap_status"].text = "Playing..."
        self._widgets["diap_status"].style = {"color": 0xFF00FF00}

    def _on_diap_pause(self):
        if not self._diap:
            return
        if self._diap.is_paused:
            self._diap.resume()
            self._widgets["diap_pause"].text = "Pause"
        else:
            self._diap.pause()
            self._widgets["diap_pause"].text = "Resume"
            self._widgets["diap_status"].text = "Paused"
            self._widgets["diap_status"].style = {"color": 0xFFFFFF00}

    def _on_diap_stop(self):
        if not self._diap:
            return
        self._diap.stop()
        self._widgets["diap_play"].text = "Play"
        self._widgets["diap_play"].style = {"background_color": 0xFF2A7F2A}
        self._widgets["diap_pause"].enabled = False
        self._widgets["diap_pause"].text = "Pause"
        self._widgets["diap_status"].text = "Stopped"
        self._widgets["diap_status"].style = {"color": 0xFFAAAAAA}
        self._widgets["diap_timeline"].model.set_value(0)

    def _on_diap_step_fwd(self):
        if self._diap:
            self._diap.step_forward()

    def _on_diap_step_back(self):
        if self._diap:
            self._diap.step_backward()

    def _on_export_clicked(self):
        pass

    def _on_connect_clicked(self):
        pass

    # ── Stats Update ──────────────────────────────────────────────────

    def update_stats(self):
        """Update statistics display — called every frame by extension."""
        if self._mode == "cell_sim":
            self._update_cell_sim_stats()
        elif self._mode == "diapedesis":
            self._update_diap_stats()

    def _update_cell_sim_stats(self):
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

    def _update_diap_stats(self):
        if not self._diap:
            return
        stats = self._diap.stats
        total = self._diap.total_frames
        cur = self._diap.current_frame

        if "diap_stat_leuko" in self._widgets:
            self._widgets["diap_stat_leuko"].text = str(stats.get("leukocytes", 0))
        if "diap_stat_rbc" in self._widgets:
            self._widgets["diap_stat_rbc"].text = str(stats.get("rbcs", 0))
        if "diap_stat_bacteria" in self._widgets:
            self._widgets["diap_stat_bacteria"].text = str(stats.get("bacteria_alive", 0))
        if "diap_stat_fps" in self._widgets:
            self._widgets["diap_stat_fps"].text = f"{stats.get('fps', 0):.1f}"
        if "diap_stat_frame_ms" in self._widgets:
            self._widgets["diap_stat_frame_ms"].text = f"{stats.get('step_time_ms', 0):.2f} ms"

        # Update timeline position
        if "diap_timeline" in self._widgets and total > 0:
            self._widgets["diap_timeline"].model.set_value(float(cur))
        if "diap_frame_lbl" in self._widgets:
            self._widgets["diap_frame_lbl"].text = f"Frame: {cur} / {total}"
        if "diap_time_lbl" in self._widgets:
            self._widgets["diap_time_lbl"].text = f"Time: {stats.get('time', 0):.1f}s"

        # Update play button if playback ended
        if not self._diap.is_playing and "diap_play" in self._widgets:
            self._widgets["diap_play"].text = "Play"
            self._widgets["diap_play"].style = {"background_color": 0xFF2A7F2A}
            if cur >= total - 1:
                self._widgets["diap_status"].text = "Playback complete"
                self._widgets["diap_status"].style = {"color": 0xFF88FF88}
