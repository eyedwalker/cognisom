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

        # Patch crashing extensions for headless streaming mode.
        # IMPORTANT: Do NOT patch get_active_viewport() — the WebRTC
        # streamer needs it to find the pixel buffer to encode.
        if self._headless:
            self._patch_headless_crashes()

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

    # ── Headless Crash Patches ───────────────────────────────────────────

    def _patch_headless_crashes(self):
        """Patch extensions that crash in headless streaming mode.

        In headless mode, several extensions register stage-change or
        per-frame listeners that assume a GUI viewport exists. When
        they access viewport internals (ViewportWindow.__viewport_layers,
        ViewportCameraState, etc.), they crash with AttributeError or
        RuntimeError, flooding the main thread and blocking HTTP responses.

        Strategy:
        - Wrap the specific crashing handler functions with try/except
        - Do NOT patch get_active_viewport() — the WebRTC streamer needs
          it to find the render target for encoding
        - Filter stderr to suppress remaining error output noise
        """
        import sys

        # ── Patch 1: omni.physx.ui on_stage_update ──
        # This handler calls ViewportCameraState() on every USD change,
        # causing "No default or provided Viewport" RuntimeError floods.
        physx_patched = False
        for mod_path in [
            "omni.physxui.scripts.extension",
            "omni.physx.ui.scripts.extension",
            "omni.physxui.extension",
        ]:
            try:
                import importlib
                physx_ext_mod = importlib.import_module(mod_path)
                # Find the Extension class (may be named differently)
                for attr_name in dir(physx_ext_mod):
                    cls = getattr(physx_ext_mod, attr_name)
                    if (isinstance(cls, type) and
                            hasattr(cls, 'on_stage_update')):
                        _orig_fn = cls.on_stage_update

                        def _safe_stage_update(self_ext, *a, _f=_orig_fn,
                                               **kw):
                            try:
                                return _f(self_ext, *a, **kw)
                            except (RuntimeError, AttributeError):
                                pass

                        cls.on_stage_update = _safe_stage_update
                        carb.log_warn(f"[cognisom.sim] Patched "
                                      f"{mod_path}.{attr_name}."
                                      f"on_stage_update")
                        physx_patched = True
                        break
                if physx_patched:
                    break
            except ImportError:
                continue
            except Exception as e:
                carb.log_warn(f"[cognisom.sim] physx patch via "
                              f"{mod_path} failed: {e}")
        if not physx_patched:
            carb.log_warn("[cognisom.sim] Could not patch physx.ui "
                          "on_stage_update — will rely on stderr filter")

        # ── Patch 2: ViewportCameraState.__init__ ──
        # Any remaining code that creates ViewportCameraState() will
        # crash in headless mode. Make the constructor safe.
        try:
            from omni.kit.viewport.utility.camera_state import \
                ViewportCameraState

            _orig_cam_init = ViewportCameraState.__init__

            def _safe_camera_init(self_cam, *args, **kwargs):
                try:
                    _orig_cam_init(self_cam, *args, **kwargs)
                except (RuntimeError, AttributeError):
                    self_cam._viewport_api = None

            ViewportCameraState.__init__ = _safe_camera_init

            # Also patch position_world/target_world to not crash
            # when _viewport_api is None
            for prop_name in ('position_world', 'target_world'):
                if hasattr(ViewportCameraState, prop_name):
                    orig_prop = getattr(ViewportCameraState, prop_name)
                    if isinstance(orig_prop, property) and orig_prop.fget:
                        _fget = orig_prop.fget

                        def _safe_fget(self_cam, _fget=_fget):
                            if getattr(self_cam, '_viewport_api', None) is None:
                                from pxr import Gf
                                return Gf.Vec3d(0, 0, 0)
                            try:
                                return _fget(self_cam)
                            except (RuntimeError, AttributeError):
                                from pxr import Gf
                                return Gf.Vec3d(0, 0, 0)

                        _fset = orig_prop.fset
                        setattr(ViewportCameraState, prop_name,
                                property(_safe_fget, _fset))

            carb.log_warn("[cognisom.sim] Patched ViewportCameraState "
                          "for headless mode")
        except ImportError:
            carb.log_info("[cognisom.sim] ViewportCameraState not available")
        except Exception as e:
            carb.log_info(f"[cognisom.sim] ViewportCameraState "
                          f"patch skipped: {e}")

        # ── Patch 3: viewport_widgets_manager._on_update ──
        # This per-frame handler crashes accessing viewport widget state.
        try:
            from omni.kit.viewport_widgets_manager import manager as vwm_mod
            if hasattr(vwm_mod, 'ViewportWidgetManager'):
                VWM = vwm_mod.ViewportWidgetManager
                if hasattr(VWM, '_on_update'):
                    _orig_vwm_update = VWM._on_update

                    def _safe_vwm_update(self_mgr, *args, **kwargs):
                        try:
                            return _orig_vwm_update(self_mgr, *args, **kwargs)
                        except (AttributeError, RuntimeError):
                            pass

                    VWM._on_update = _safe_vwm_update
                    carb.log_warn("[cognisom.sim] Patched "
                                  "viewport_widgets_manager for headless")
        except ImportError:
            pass
        except Exception as e:
            carb.log_info(f"[cognisom.sim] widgets_manager patch skipped: {e}")

        # ── Patch 4: ViewportWindow.viewport_api property ──
        # Prevent AttributeError on __viewport_layers access
        try:
            from omni.kit.viewport.window import ViewportWindow
            if hasattr(ViewportWindow, 'viewport_api'):
                orig_prop = ViewportWindow.viewport_api
                if isinstance(orig_prop, property) and orig_prop.fget:
                    _orig_fget = orig_prop.fget

                    def safe_viewport_api(self_win):
                        try:
                            return _orig_fget(self_win)
                        except AttributeError:
                            return None

                    ViewportWindow.viewport_api = property(safe_viewport_api)
                    carb.log_warn("[cognisom.sim] Patched "
                                  "ViewportWindow.viewport_api")
        except ImportError:
            pass
        except Exception as e:
            carb.log_info(f"[cognisom.sim] ViewportWindow patch skipped: {e}")

        # ── Patch 5: Filter stderr for remaining error noise ──
        _orig_stderr_write = sys.stderr.write
        _suppress_count = [0]

        def filtered_stderr_write(text):
            if 'viewport_layers' in text or 'viewport_api' in text:
                _suppress_count[0] += 1
                return len(text)
            if 'No default or provided Viewport' in text:
                _suppress_count[0] += 1
                return len(text)
            if 'ViewportCameraState' in text:
                _suppress_count[0] += 1
                return len(text)
            if ('omni.kit.viewport' in text and
                    ('AttributeError' in text or 'RuntimeError' in text)):
                _suppress_count[0] += 1
                return len(text)
            if ('physxui' in text and 'on_stage_update' in text):
                _suppress_count[0] += 1
                return len(text)
            if ('camera_state' in text and 'AttributeError' in text):
                _suppress_count[0] += 1
                return len(text)
            if ('viewport_widgets_manager' in text and
                    'AttributeError' in text):
                _suppress_count[0] += 1
                return len(text)
            if '__legacy_window' in text:
                _suppress_count[0] += 1
                return len(text)
            return _orig_stderr_write(text)

        sys.stderr.write = filtered_stderr_write

        carb.log_warn("[cognisom.sim] Headless crash patches installed")

    # ── Headless Scene Setup ──────────────────────────────────────────────

    async def _setup_headless_scene(self):
        """Set up camera and scene for headless WebRTC streaming.

        With isaacsim.exp.full.streaming, the WebRTC streamer
        (omni.kit.livestream.webrtc) handles RTX rendering and streaming
        directly. We set up:
        1. USD stage with wait-for-ready
        2. Camera with good viewing angle for diapedesis scene
        3. RTX 2.0 settings for bio-specific rendering
        4. Explicit render product for the camera
        5. Bind render product to livestream if possible
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

        # Create explicit render product for this camera
        rp_path = await self._create_render_product(camera_path)

        # Check if WebRTC streaming is available
        webrtc_available = self._check_webrtc_streaming()

        if webrtc_available:
            carb.log_warn("[cognisom.sim] WebRTC streaming active.")
            if rp_path:
                self._bind_render_product_to_webrtc(rp_path)
        else:
            carb.log_warn("[cognisom.sim] No WebRTC streaming, "
                          "trying replicator capture for MJPEG...")
            await self._setup_rtx_capture(camera_path)

        carb.log_warn("[cognisom.sim] Headless scene setup complete — "
                      "HTTP API on port 8211")

    async def _create_render_product(self, camera_path: str):
        """Create an explicit render product for the camera.

        This render product can be bound to the WebRTC streamer or used
        for MJPEG capture. It bypasses the UI viewport layer entirely.
        """
        app = omni.kit.app.get_app()
        try:
            import omni.replicator.core as rep
            carb.log_warn(f"[cognisom.sim] Creating render product for "
                          f"camera: {camera_path}")
            rp = rep.create.render_product(camera_path, (1920, 1080))
            self._render_product = rp

            # Wait for render product to initialize
            for _ in range(10):
                await app.next_update_async()

            # Get the render product prim path
            rp_path = None
            if hasattr(rp, 'path'):
                rp_path = rp.path
            elif isinstance(rp, str):
                rp_path = rp

            carb.log_warn(f"[cognisom.sim] Render product created: {rp_path}")
            return rp_path
        except Exception as e:
            carb.log_warn(f"[cognisom.sim] Could not create render product: {e}")
            return None

    def _bind_render_product_to_webrtc(self, rp_path: str):
        """Bind our render product to the WebRTC livestream."""
        # Try multiple APIs for different Kit versions
        bound = False

        # Method 1: omni.kit.livestream.core (Kit 105+)
        try:
            import omni.kit.livestream.core as livestream_core
            if hasattr(livestream_core, 'set_render_product'):
                livestream_core.set_render_product(rp_path)
                carb.log_warn(f"[cognisom.sim] WebRTC bound to {rp_path} "
                              f"via livestream.core")
                bound = True
        except ImportError:
            pass
        except Exception as e:
            carb.log_info(f"[cognisom.sim] livestream.core binding failed: {e}")

        # Method 2: carb settings
        if not bound:
            try:
                settings = carb.settings.get_settings()
                settings.set_string("/app/livestream/renderProduct", rp_path)
                carb.log_warn(f"[cognisom.sim] Set livestream render product "
                              f"via settings: {rp_path}")
                bound = True
            except Exception as e:
                carb.log_info(f"[cognisom.sim] Settings binding failed: {e}")

        if not bound:
            carb.log_warn("[cognisom.sim] Could not bind render product to "
                          "WebRTC — it may use the default viewport instead")

    def _check_webrtc_streaming(self) -> bool:
        """Check if the Kit WebRTC livestream extension is active."""
        # Method 1: Check via extension manager
        try:
            import omni.ext
            manager = omni.ext.get_extension_manager()
            for ext_id in ["omni.kit.livestream.webrtc",
                           "omni.services.livestream.webrtc"]:
                try:
                    if manager.is_extension_enabled(ext_id):
                        carb.log_warn(f"[cognisom.sim] Found active: {ext_id}")
                        return True
                except Exception:
                    pass
            # Try iterating enabled extensions
            try:
                for ext_info in manager.get_extensions():
                    name = ext_info.get("id", "") or ext_info.get("name", "")
                    if "livestream" in name.lower() and "webrtc" in name.lower():
                        enabled = ext_info.get("enabled", False)
                        carb.log_warn(f"[cognisom.sim] Found extension: "
                                      f"{name} enabled={enabled}")
                        if enabled:
                            return True
            except Exception:
                pass
        except Exception as e:
            carb.log_warn(f"[cognisom.sim] Extension manager check failed: {e}")

        # Method 2: Check carb settings for livestream
        try:
            settings = carb.settings.get_settings()
            port = settings.get_as_int("/app/livestream/port")
            if port and port > 0:
                carb.log_warn(f"[cognisom.sim] Livestream port set: {port}")
                return True
        except Exception:
            pass

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
        #   Scene center ~ (100, -15, 0)
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
        pitch = math.degrees(math.atan2(-dy, dist_xz))  # ~22.6 down
        yaw = math.degrees(math.atan2(dx, -dz))  # 0

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
                    try:
                        viewport_api.set_active_camera(camera_path)
                    except Exception:
                        pass
                carb.log_warn(f"[cognisom.sim] Active camera: {camera_path}")

                # Set render resolution
                try:
                    viewport_api.set_texture_resolution((1920, 1080))
                except (AttributeError, Exception):
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

        # Disable ground grid and helpers
        settings.set_bool("/app/viewport/grid/enabled", False)
        settings.set_bool("/persistent/app/viewport/displayOptions/grid", False)
        settings.set_bool("/app/viewport/show/grid", False)
        settings.set_bool("/app/viewport/show/groundPlane", False)

        # Dark background (deep space)
        settings.set_float_array("/rtx/post/tonemap/backgroundDefaultColor",
                                 [0.02, 0.02, 0.04])

        # RTX Real-Time 2.0 (RT2)
        settings.set_string("/rtx/rendermode", "RTX-Realtime")
        settings.set_bool("/rtx/rt2/enabled", True)
        carb.log_info("[cognisom.sim] RTX-Realtime mode + RT2 enabled")

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

        # Depth of field off by default
        settings.set_bool("/rtx/post/dof/enabled", False)

        # Anti-aliasing
        settings.set_bool("/rtx/post/aa/enabled", True)

        # Higher sample count for quality
        settings.set_int("/rtx/pathtracing/totalSpp", 64)
        settings.set_int("/rtx/directLighting/sampledLighting/autoNumberOfLights", 8)

        carb.log_info("[cognisom.sim] Bio-specific RTX settings applied "
                      "(translucency, GI, SSS, AO)")

    async def _setup_rtx_capture(self, camera_path: str):
        """Set up RTX frame capture for MJPEG fallback streaming."""
        app = omni.kit.app.get_app()

        # Use the render product we already created
        if self._render_product:
            try:
                import omni.replicator.core as rep
                rgb = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb.attach([self._render_product])

                # Warm up
                carb.log_warn("[cognisom.sim] Warming up MJPEG capture...")
                for _ in range(30):
                    await app.next_update_async()

                # Try stepping the orchestrator for headless
                try:
                    rep.orchestrator.run()
                except Exception:
                    pass

                if self._streaming_server:
                    self._streaming_server.set_annotator(rgb)
                self._rgb_annotator = rgb
                carb.log_warn("[cognisom.sim] MJPEG capture via "
                              "render product initialized")
                return
            except Exception as e:
                carb.log_warn(f"[cognisom.sim] Render product capture "
                              f"failed: {e}")

        # Fallback: viewport capture
        self._setup_viewport_capture()

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
        # so MJPEG clients get updated frames.
        # Rate-limit to ~10fps to avoid starving the GIL — PIL rendering
        # is CPU-intensive and blocks the main thread if called every tick.
        if (self._headless and self._streaming_server
                and not self._rgb_annotator):
            vc = self._streaming_server._viewport_capture
            if not vc._rtx_active:
                import time as _time
                now = _time.monotonic()
                if not hasattr(self, '_last_pil_capture'):
                    self._last_pil_capture = 0.0
                if now - self._last_pil_capture >= 0.1:  # 10fps cap
                    vc.capture()
                    self._last_pil_capture = now

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
