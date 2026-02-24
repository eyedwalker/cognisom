"""
Cognisom Simulation Extension
=============================

Main extension class for Omniverse Kit integration.
Supports two modes:
  - Cell Simulation: Generic tumor microenvironment (SimulationManager)
  - Diapedesis: Leukocyte extravasation cascade playback (DiapedesisManager)

Headless streaming (WebRTC):
  When running with --no-window (headless + isaacsim.exp.full.streaming),
  the Kit WebRTC streamer handles RTX rendering and streaming automatically.
  This extension sets up the camera, scene, and simulation control via HTTP.
"""

import asyncio

import carb
import omni.ext
import omni.usd

from .simulation_manager import SimulationManager
from .diapedesis_manager import DiapedesisManager
from .streaming_server import StreamingServer, _encode_jpeg


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
        self._headless = False
        self._render_product = None
        self._rgb_annotator = None
        self._viewport_api = None

    def on_startup(self, ext_id: str):
        """Called when extension is loaded."""
        carb.log_info("[cognisom.sim] Starting Cognisom Simulation Extension")

        # Get extension settings
        settings = carb.settings.get_settings()
        self._fps = settings.get_as_float("exts/cognisom.sim/simulation/fps") or 60.0
        self._dt = settings.get_as_float("exts/cognisom.sim/simulation/dt") or 0.01

        # Detect headless mode (--no-window)
        window_enabled = settings.get_as_bool("/app/window/enabled")
        self._headless = window_enabled is False or window_enabled is None

        carb.log_info(f"[cognisom.sim] Headless mode: {self._headless}")

        # Initialize both managers
        self._simulation_manager = SimulationManager()
        self._diapedesis_manager = DiapedesisManager()

        # Start HTTP server for simulation control + MJPEG fallback (port 8211)
        self._streaming_server = StreamingServer(
            self._diapedesis_manager, port=8211)
        self._streaming_server.start()

        if self._headless:
            # Headless: set up camera and scene for WebRTC streaming
            # The Kit WebRTC streamer (omni.kit.livestream.webrtc) handles
            # RTX rendering and streaming automatically via port 8899
            carb.log_info("[cognisom.sim] Setting up headless scene for WebRTC...")
            asyncio.ensure_future(self._setup_headless_scene())
        else:
            # GUI mode: create UI panels and menus
            self._create_ui()
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

    # ── Headless Scene Setup ──────────────────────────────────────────────

    async def _setup_headless_scene(self):
        """Set up camera and scene for headless WebRTC streaming.

        With isaacsim.exp.full.streaming, the WebRTC streamer
        (omni.kit.livestream.webrtc) handles RTX rendering and streaming
        directly — no replicator capture needed. We set up:
        1. USD stage with wait-for-ready
        2. Camera with good viewing angle for diapedesis scene
        3. RTX 2.0 settings for bio-specific rendering
        4. The PIL fallback renderer handles MJPEG for non-WebRTC clients
        """
        app = omni.kit.app.get_app()

        # Wait for Kit to fully initialize (stage may take time to be ready)
        carb.log_warn("[cognisom.sim] Waiting for Kit initialization...")
        context = omni.usd.get_context()
        stage = None
        for attempt in range(60):  # Up to ~2 seconds
            await app.next_update_async()
            stage = context.get_stage()
            if stage:
                break
            if attempt == 15:
                carb.log_warn("[cognisom.sim] Still waiting for stage...")

        if not stage:
            carb.log_warn("[cognisom.sim] No stage after init, creating new...")
            result, error = await context.new_stage_async()
            if not result:
                carb.log_error(f"[cognisom.sim] Failed to create stage: {error}")
                return
            # Wait for new stage to be ready
            for attempt in range(30):
                await app.next_update_async()
                stage = context.get_stage()
                if stage:
                    break

        if not stage:
            carb.log_error("[cognisom.sim] Stage still not available after creation")
            return

        carb.log_warn(f"[cognisom.sim] USD stage ready: "
                      f"{stage.GetRootLayer().identifier}")

        # Update diapedesis manager to use the Kit context stage
        if self._diapedesis_manager:
            self._diapedesis_manager._stage = stage
            carb.log_warn("[cognisom.sim] DiapedesisManager stage updated")

        # Create camera for the diapedesis scene
        camera_path = self._create_scene_camera(stage)

        # Wait for prims to propagate
        for _ in range(3):
            await app.next_update_async()

        # Set camera as active viewport camera (best effort — may fail headless)
        self._set_active_camera(camera_path)

        # Apply RTX 2.0 + bio-specific rendering settings
        self._apply_viewport_settings()

        # Wait for renderer to pick up settings
        for _ in range(10):
            await app.next_update_async()

        # Check if WebRTC streaming is available
        webrtc_available = self._check_webrtc_streaming()

        if webrtc_available:
            carb.log_warn("[cognisom.sim] WebRTC streaming active — "
                          "RTX frames streamed directly via WebRTC. "
                          "MJPEG fallback uses PIL 2D renderer.")
        else:
            # Only try replicator capture if WebRTC is not available
            carb.log_warn("[cognisom.sim] No WebRTC streaming, "
                          "trying replicator capture for MJPEG...")
            await self._setup_rtx_capture(camera_path)

        carb.log_warn("[cognisom.sim] Headless scene setup complete — "
                      "HTTP API on port 8211")

    def _check_webrtc_streaming(self) -> bool:
        """Check if the Kit WebRTC livestream extension is active."""
        try:
            import omni.ext
            manager = omni.ext.get_extension_manager()
            # Check for the WebRTC livestream extension
            for ext_id in ["omni.kit.livestream.webrtc",
                           "omni.services.livestream.webrtc"]:
                if manager.is_extension_enabled(ext_id):
                    carb.log_warn(f"[cognisom.sim] Found active: {ext_id}")
                    return True
        except Exception as e:
            carb.log_info(f"[cognisom.sim] Extension check failed: {e}")
        return False

    def _create_scene_camera(self, stage) -> str:
        """Create and position a camera to view the diapedesis scene."""
        from pxr import UsdGeom, Gf

        camera_path = "/World/DiapedesisCam"

        # Remove existing camera if present
        old = stage.GetPrimAtPath(camera_path)
        if old and old.IsValid():
            stage.RemovePrim(camera_path)

        camera_prim = stage.DefinePrim(camera_path, "Camera")
        camera = UsdGeom.Camera(camera_prim)
        camera.GetFocalLengthAttr().Set(24.0)  # Moderate wide angle
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))
        camera.GetHorizontalApertureAttr().Set(36.0)

        # Scene layout:
        #   Vessel: cylinder along X from 0→200, radius 25, center at Y=0
        #   Tissue: below vessel, Y from -25 to -75
        #   Scene center ≈ (100, -15, 0)
        #
        # Camera: elevated 3/4 view from Z+, looking down at the vessel
        cam_pos = Gf.Vec3d(100.0, 60.0, 180.0)
        target = Gf.Vec3d(100.0, -15.0, 0.0)

        xformable = UsdGeom.Xformable(camera_prim)
        xformable.AddTranslateOp().Set(cam_pos)

        # Compute rotation to look at vessel center
        import math
        dx = target[0] - cam_pos[0]  # 0
        dy = target[1] - cam_pos[1]  # -75
        dz = target[2] - cam_pos[2]  # -180
        dist_xz = math.sqrt(dx * dx + dz * dz)  # 180
        pitch = math.degrees(math.atan2(-dy, dist_xz))  # ~22.6° down
        yaw = math.degrees(math.atan2(dx, -dz))  # 0°

        xformable.AddRotateXYZOp().Set(Gf.Vec3f(-pitch, yaw, 0.0))

        carb.log_info(f"[cognisom.sim] Camera at {camera_path} "
                      f"pos={cam_pos} target={target} pitch={pitch:.1f}")
        return camera_path

    def _set_active_camera(self, camera_path: str):
        """Set the active viewport camera with proper resolution."""
        try:
            import omni.kit.viewport.utility as vp_util
            viewport_api = vp_util.get_active_viewport()
            if viewport_api:
                # Set camera path (preferred API)
                try:
                    viewport_api.camera_path = camera_path
                except AttributeError:
                    viewport_api.set_active_camera(camera_path)
                carb.log_info(f"[cognisom.sim] Active camera set to {camera_path}")

                # Set render resolution
                try:
                    viewport_api.set_texture_resolution((1920, 1080))
                    carb.log_info("[cognisom.sim] Viewport texture resolution set to 1920x1080")
                except AttributeError:
                    try:
                        viewport_api.resolution = (1920, 1080)
                    except Exception:
                        pass

                # Store viewport reference for later use
                self._viewport_api = viewport_api
                return
        except Exception as e:
            carb.log_info(f"[cognisom.sim] Could not set active camera "
                          f"via viewport utility: {e}")

        # Fallback: set via settings
        try:
            settings = carb.settings.get_settings()
            settings.set_string("/app/viewport/defaultCamera", camera_path)
            carb.log_info(f"[cognisom.sim] Default camera set via settings: "
                          f"{camera_path}")
        except Exception as e:
            carb.log_warn(f"[cognisom.sim] Could not set default camera: {e}")

    def _apply_viewport_settings(self):
        """Apply visual settings for the biological scene.

        Enables RTX Real-Time 2.0 (RT2) with bio-specific rendering:
        - Translucency for cell membranes and tissue
        - Indirect diffuse (GI) for soft tissue lighting
        - Subsurface scattering for organic materials
        - High-quality shadows and reflections
        """
        settings = carb.settings.get_settings()

        # ── Disable ground grid and helpers ──
        settings.set_bool("/app/viewport/grid/enabled", False)
        settings.set_bool("/persistent/app/viewport/displayOptions/grid", False)
        settings.set_bool("/app/viewport/show/grid", False)
        settings.set_bool("/app/viewport/show/groundPlane", False)

        # ── Dark background (deep space) ──
        settings.set_float_array("/rtx/post/tonemap/backgroundDefaultColor",
                                 [0.02, 0.02, 0.04])

        # ── RTX Real-Time 2.0 (RT2) ──
        settings.set_string("/rtx/rendermode", "RTX-Realtime")
        settings.set_bool("/rtx/rt2/enabled", True)
        carb.log_info("[cognisom.sim] RTX-Realtime mode + RT2 enabled")

        # ── Bio-specific rendering fidelity ──
        # Translucency: critical for cell membranes, tissue sections
        settings.set_bool("/rtx/translucency/enabled", True)
        settings.set_int("/rtx/translucency/maxRefractionBounces", 4)

        # Indirect diffuse (GI): soft fill light through tissue
        settings.set_bool("/rtx/indirectDiffuse/enabled", True)

        # Subsurface scattering: organic materials (cell bodies, tissue)
        settings.set_bool("/rtx/subsurface/enabled", True)

        # Ambient occlusion: depth cues in vessel interior
        settings.set_bool("/rtx/post/aa/op", True)
        settings.set_bool("/rtx/ambientOcclusion/enabled", True)

        # Depth of field: subtle focus on vessel center
        settings.set_bool("/rtx/post/dof/enabled", False)  # Off by default

        # Anti-aliasing
        settings.set_bool("/rtx/post/aa/enabled", True)

        # Higher sample count for quality (reduce noise in translucent areas)
        settings.set_int("/rtx/pathtracing/totalSpp", 64)
        settings.set_int("/rtx/directLighting/sampledLighting/autoNumberOfLights", 8)

        carb.log_info("[cognisom.sim] Bio-specific RTX settings applied "
                      "(translucency, GI, SSS, AO)")

    async def _setup_rtx_capture(self, camera_path: str):
        """Set up RTX frame capture for MJPEG fallback streaming.

        Tries three approaches in order:
        1. Viewport render product (uses the existing streaming viewport)
        2. Replicator render product (creates a new one from camera)
        3. Viewport API capture callback

        The render pipeline needs significant warm-up time (50+ frames)
        before the annotator produces data.
        """
        app = omni.kit.app.get_app()

        # ── Approach 1: Use viewport's existing render product ──
        try:
            import omni.kit.viewport.utility as vp_util
            viewport_api = vp_util.get_active_viewport()
            if viewport_api:
                rp_path = viewport_api.render_product_path
                if rp_path:
                    carb.log_info(f"[cognisom.sim] Viewport render product: {rp_path}")
                    import omni.replicator.core as rep
                    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
                    rgb.attach([rp_path])

                    # Extended warm-up: renderer needs time to produce pixels
                    carb.log_info("[cognisom.sim] Warming up renderer (50 frames)...")
                    for i in range(50):
                        await app.next_update_async()
                        if i == 25:
                            # Check mid-warm-up
                            import numpy as np
                            test = rgb.get_data()
                            if test is not None:
                                arr = np.array(test)
                                if arr.size > 100:
                                    carb.log_info(
                                        f"[cognisom.sim] Render product active at "
                                        f"frame {i}: shape={arr.shape}")
                                    break

                    if self._streaming_server:
                        self._streaming_server.set_annotator(rgb)
                    self._rgb_annotator = rgb
                    carb.log_warn("[cognisom.sim] RTX capture via viewport "
                                  "render product initialized")
                    return
        except Exception as e:
            carb.log_info(f"[cognisom.sim] Viewport render product approach "
                          f"failed: {e}")

        # ── Approach 2: Create new replicator render product ──
        try:
            import omni.replicator.core as rep

            rp = rep.create.render_product(camera_path, (1920, 1080))

            # Extended warm-up
            carb.log_info("[cognisom.sim] Warming up new render product (60 frames)...")
            for _ in range(30):
                await app.next_update_async()

            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach([rp])

            for _ in range(30):
                await app.next_update_async()

            if self._streaming_server:
                self._streaming_server.set_annotator(rgb)

            self._render_product = rp
            self._rgb_annotator = rgb
            carb.log_warn("[cognisom.sim] RTX capture via new replicator "
                          "render product initialized")
            return

        except ImportError:
            carb.log_info("[cognisom.sim] omni.replicator.core not available")
        except Exception as e:
            carb.log_info(f"[cognisom.sim] Replicator capture failed: {e}")

        # ── Approach 3: Viewport API fallback ──
        self._setup_viewport_capture()
        carb.log_warn("[cognisom.sim] Using viewport API capture fallback")

    def _setup_viewport_capture(self):
        """Fallback: set viewport API on streaming server for frame capture."""
        try:
            import omni.kit.viewport.utility as vp_util
            viewport_api = vp_util.get_active_viewport()
            if viewport_api and self._streaming_server:
                self._streaming_server.set_viewport(viewport_api)
                carb.log_warn("[cognisom.sim] Viewport API capture configured")
        except Exception as e:
            carb.log_warn(f"[cognisom.sim] Viewport capture also failed: {e}")

    # ── GUI Mode ─────────────────────────────────────────────────────────

    def _create_ui(self):
        """Create the extension UI window (GUI mode only)."""
        import omni.ui as ui
        from .ui_panel import CognisomPanel

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
        """Add extension to Omniverse menu (GUI mode only)."""
        try:
            editor_menu = omni.kit.ui.get_editor_menu()

            menu_item = editor_menu.add_item(
                "Window/Simulation/Cognisom",
                self._on_menu_click,
                toggle=True,
                value=True
            )
            self._menu_items.append(menu_item)

            diap_item = editor_menu.add_item(
                "Window/Simulation/Cognisom Diapedesis",
                self._on_diapedesis_menu_click,
            )
            self._menu_items.append(diap_item)
        except Exception as e:
            carb.log_info(f"[cognisom.sim] Menu creation skipped: {e}")

    def _on_menu_click(self, menu_item, value):
        if self._window:
            self._window.visible = value

    def _on_diapedesis_menu_click(self, menu_item, value):
        self._mode = "diapedesis"
        if self._panel:
            self._panel.set_mode("diapedesis")
        if self._window:
            self._window.visible = True

    def _on_mode_changed(self, mode: str):
        self._mode = mode
        carb.log_info(f"[cognisom.sim] Mode changed to: {mode}")

    # ── Per-Frame Update ─────────────────────────────────────────────────

    def _on_update(self, event):
        """Called every frame on Kit main thread."""
        dt = event.payload.get("dt", 1.0 / 60.0)

        # Process any queued actions from HTTP threads (must run on main thread)
        if self._diapedesis_manager:
            self._diapedesis_manager.process_pending()

        if self._mode == "cell_sim":
            if self._simulation_manager and self._simulation_manager.is_running:
                self._simulation_manager.update(dt)

        # In headless mode, always drive diapedesis from main thread
        if self._headless:
            if (self._diapedesis_manager and
                    self._diapedesis_manager.is_playing):
                self._diapedesis_manager.update(dt)
        elif self._mode == "diapedesis":
            if self._diapedesis_manager and self._diapedesis_manager.is_playing:
                self._diapedesis_manager.update(dt)

        # Capture RTX frame from replicator annotator (only when NOT using WebRTC)
        if self._rgb_annotator and self._streaming_server:
            self._capture_rtx_frame()

        # In headless mode with PIL fallback active, trigger a capture
        # so MJPEG clients get updated frames
        if (self._headless and self._streaming_server
                and not self._rgb_annotator):
            vc = self._streaming_server._viewport_capture
            if not vc._rtx_active:
                vc.capture()

        # Update UI (GUI mode only)
        if self._panel:
            self._panel.update_stats()

    def _capture_rtx_frame(self):
        """Read RTX frame from replicator annotator and push to streaming buffer.

        Called every frame on Kit main thread. When the annotator produces
        valid pixel data, encodes it as JPEG and pushes to the viewport
        capture buffer for MJPEG serving.
        """
        try:
            import numpy as np

            data = self._rgb_annotator.get_data()

            # Diagnostic logging (throttled)
            if not hasattr(self, '_diag_count'):
                self._diag_count = 0
            self._diag_count += 1
            if self._diag_count <= 5 or self._diag_count % 600 == 0:
                log_fn = carb.log_warn if self._diag_count <= 5 else carb.log_info
                if data is not None:
                    arr_tmp = np.array(data)
                    log_fn(
                        f"[cognisom.sim] Annotator: shape={arr_tmp.shape} "
                        f"size={arr_tmp.size} (frame {self._diag_count})")
                else:
                    log_fn(
                        f"[cognisom.sim] Annotator: None (frame {self._diag_count})")

            if data is None:
                return

            arr = np.array(data)
            if arr.size == 0 or arr.ndim < 2:
                return

            h, w = arr.shape[:2]
            if h < 10 or w < 10:
                return

            channels = arr.shape[2] if arr.ndim == 3 else 1
            if channels < 3:
                return

            # Encode RGB to JPEG
            rgb = arr[:, :, :3]
            jpeg_bytes = _encode_jpeg(rgb.tobytes(), w, h)
            if jpeg_bytes and len(jpeg_bytes) > 1000:
                vc = self._streaming_server._viewport_capture
                vc._buffer = jpeg_bytes
                vc._width = w
                vc._height = h
                if not vc._rtx_active:
                    vc._rtx_active = True
                    carb.log_warn(
                        f"[cognisom.sim] RTX capture active! {w}x{h}, "
                        f"{len(jpeg_bytes)//1024}KB JPEG")
        except Exception as e:
            if not hasattr(self, '_rtx_error_count'):
                self._rtx_error_count = 0
            self._rtx_error_count += 1
            if self._rtx_error_count <= 3:
                carb.log_warn(f"[cognisom.sim] RTX capture error: {e}")


# Extension entry point
def get_extension():
    """Return extension instance."""
    return CognisomSimExtension()
