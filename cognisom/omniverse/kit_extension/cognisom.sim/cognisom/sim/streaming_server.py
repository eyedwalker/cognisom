"""
HTTP Streaming Server for Kit Viewport
=======================================

Serves RTX-rendered viewport frames as JPEG over HTTP.
This replaces WebRTC streaming with a simpler, firewall-friendly approach.

Endpoints:
    GET  /status                          → JSON health check
    GET  /frame.jpg                       → Current viewport as JPEG
    GET  /stream                          → MJPEG stream (multipart)
    POST /diapedesis                      → Load frames JSON, build scene
    POST /diapedesis/play                 → Start playback
    POST /diapedesis/pause                → Pause playback
    POST /diapedesis/stop                 → Stop playback
    POST /diapedesis/seek?frame=N         → Seek to frame N

Usage from extension.py::

    server = StreamingServer(diapedesis_manager, port=8211)
    server.start()           # Non-blocking, runs in daemon thread
    server.shutdown()        # Clean stop
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

import carb

log = logging.getLogger(__name__)

# Try importing viewport capture utilities
try:
    import omni.kit.viewport.utility as vp_util
    VIEWPORT_AVAILABLE = True
except ImportError:
    VIEWPORT_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False


def _encode_jpeg(rgba_buffer, width: int, height: int, quality: int = 85) -> bytes:
    """Encode RGBA buffer to JPEG bytes."""
    if not NP_AVAILABLE:
        return b""

    try:
        # Try Pillow first (fast)
        from PIL import Image
        arr = np.frombuffer(rgba_buffer, dtype=np.uint8).reshape(height, width, 4)
        img = Image.fromarray(arr[:, :, :3])  # Drop alpha
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except ImportError:
        pass

    try:
        # Fallback: turbo-jpeg or raw BMP
        import struct
        arr = np.frombuffer(rgba_buffer, dtype=np.uint8).reshape(height, width, 4)
        rgb = arr[::-1, :, :3].tobytes()  # Flip vertical, drop alpha
        # Minimal BMP (uncompressed)
        row_size = width * 3
        padding = (4 - row_size % 4) % 4
        padded_row = row_size + padding
        file_size = 54 + padded_row * height
        bmp = struct.pack(
            "<2sIHHI IIIHHIIIIII",
            b"BM", file_size, 0, 0, 54,
            40, width, height, 1, 24, 0, padded_row * height, 2835, 2835, 0, 0,
        )
        return bmp + rgb
    except Exception:
        return b""


class _ViewportCapture:
    """Captures Kit viewport or renders simulation frames via PIL fallback.

    Supports three capture modes (tried in order):
    1. Viewport API: capture from Kit viewport (set_viewport)
    2. Replicator annotator: read from render product (set_annotator)
    3. PIL fallback: 2D scientific visualization from simulation data
    """

    # Leukocyte state colors (matches Three.js viewer)
    STATE_COLORS = {
        0: (100, 200, 255),   # FLOWING — light blue
        1: (255, 220, 50),    # TETHERED — yellow
        2: (255, 165, 0),     # ROLLING — orange
        3: (0, 255, 200),     # ACTIVATING — cyan
        4: (255, 50, 50),     # ARRESTED — red
        5: (180, 0, 255),     # TRANSMIGRATING — purple
        6: (0, 255, 80),      # EXTRAVASATED — green
    }

    def __init__(self):
        self._buffer: Optional[bytes] = None
        self._width = 1920
        self._height = 1080
        self._last_capture_time = 0.0
        self._diapedesis_mgr = None  # Set by StreamingServer
        self._viewport_api = None    # Set by extension (approach 1 or 2)
        self._rgb_annotator = None   # Set by extension (approach 3)
        self._rtx_active = False     # True once RTX frames are confirmed
        self._capture_attempt_count = 0  # Throttle error logging

    @property
    def jpeg_bytes(self) -> Optional[bytes]:
        return self._buffer

    def capture(self) -> bool:
        """Capture current frame.

        Viewport capture is driven from Kit's main thread (_on_update in
        extension.py) because schedule_capture needs an asyncio event loop.
        This method is called from the daemon capture thread and handles:
        - Replicator annotator capture (if configured)
        - PIL fallback (if no RTX buffer is available yet)

        When RTX capture is active (main thread writing to _buffer), this
        method returns True without doing anything extra.
        """
        # If RTX capture is active (main thread updating buffer), just return
        if self._rtx_active and self._buffer:
            return True

        # ── Replicator annotator capture ──
        if self._rgb_annotator:
            result = self._capture_from_annotator()
            if result:
                if not self._rtx_active:
                    carb.log_info("[streaming] Replicator capture active!")
                    self._rtx_active = True
                return True

        # ── Fallback: PIL ──
        return self._render_sim_frame()

    def _capture_from_viewport(self) -> bool:
        """Capture frame from Kit viewport API."""
        try:
            from omni.kit.viewport.utility import capture_viewport_to_buffer

            viewport_api = self._viewport_api
            res = viewport_api.resolution
            if res[0] > 0 and res[1] > 0:
                self._width = res[0]
                self._height = res[1]

            captured = [False]
            buffer_holder = [None]
            dims = [0, 0]

            def on_done(buf, buf_size, w, h, fmt):
                if buf:
                    buffer_holder[0] = bytes(buf)
                    dims[0] = w
                    dims[1] = h
                    captured[0] = True

            capture_viewport_to_buffer(viewport_api, on_done)

            # Wait for capture callback (up to 200ms)
            for _ in range(20):
                if captured[0]:
                    break
                time.sleep(0.01)

            if buffer_holder[0]:
                w = dims[0] if dims[0] > 0 else self._width
                h = dims[1] if dims[1] > 0 else self._height
                self._buffer = _encode_jpeg(buffer_holder[0], w, h)
                self._last_capture_time = time.time()
                return True
        except Exception as e:
            self._capture_attempt_count += 1
            if self._capture_attempt_count <= 3:
                carb.log_info(f"[streaming] Viewport capture attempt {self._capture_attempt_count}: {e}")
        return False

    def _capture_from_annotator(self) -> bool:
        """Capture frame from replicator RGB annotator."""
        try:
            data = self._rgb_annotator.get_data()
            if data is not None and NP_AVAILABLE:
                import numpy as np
                arr = np.array(data)
                if arr.size > 0 and arr.ndim >= 2:
                    h, w = arr.shape[:2]
                    self._width = w
                    self._height = h
                    # Annotator returns RGBA or RGB
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        rgba = arr[:, :, :4] if arr.shape[2] >= 4 else arr
                        self._buffer = _encode_jpeg(
                            rgba.tobytes(), w, h)
                        self._last_capture_time = time.time()
                        return True
        except Exception as e:
            self._capture_attempt_count += 1
            if self._capture_attempt_count <= 3:
                carb.log_info(f"[streaming] Annotator capture attempt {self._capture_attempt_count}: {e}")
        return False

    def _render_sim_frame(self) -> bool:
        """Render current simulation frame as a 2D scientific visualization."""
        mgr = self._diapedesis_mgr
        if not mgr or not mgr._frames or mgr.current_frame >= len(mgr._frames):
            return False

        try:
            from PIL import Image, ImageDraw, ImageFont
            import math
        except ImportError:
            return False

        frame = mgr._frames[mgr.current_frame]
        W, H = self._width, self._height
        img = Image.new("RGB", (W, H), (8, 8, 20))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
            font_sm = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
            font_lg = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
        except Exception:
            font = font_sm = font_lg = ImageFont.load_default()

        R = frame.get("vessel_radius", 25.0)
        L = frame.get("vessel_length", 200.0)

        # Coordinate transform: simulation → pixel
        margin_x, margin_y = 60, 80
        sx = (W - 2 * margin_x) / L
        sy = (H - 2 * margin_y) / (R * 3.5)
        cx = margin_x
        cy = H * 0.35  # Vessel center (upper portion of image)

        def sim_to_px(x, y):
            return int(cx + x * sx), int(cy - y * sy)

        # ── Background: tissue space gradient ──
        for row in range(int(cy + R * sy), H):
            t = min(1.0, (row - cy - R * sy) / (H - cy - R * sy + 1))
            r = int(15 + t * 10)
            g = int(12 + t * 5)
            b = int(25 + t * 8)
            draw.line([(0, row), (W, row)], fill=(r, g, b))

        # ── Vessel walls (top and bottom arcs) ──
        wall_color = (180, 100, 120)
        wall_inflamed = (220, 60, 80)
        pts_top = []
        pts_bot = []
        for i in range(50):
            x = (i / 49) * L
            px_x = cx + x * sx
            pts_top.append((px_x, cy - R * sy))
            pts_bot.append((px_x, cy + R * sy))

        # Draw vessel wall bands
        for i in range(len(pts_top) - 1):
            selexpr = frame.get("endo_selectin_expr", [0.0])
            idx = min(i * len(selexpr) // 50, len(selexpr) - 1)
            se = selexpr[idx] if selexpr else 0.0
            c = tuple(int(wall_color[j] + (wall_inflamed[j] - wall_color[j]) * se)
                      for j in range(3))
            # Top wall
            draw.line([pts_top[i], pts_top[i + 1]], fill=c, width=3)
            # Bottom wall
            draw.line([pts_bot[i], pts_bot[i + 1]], fill=c, width=3)

        # ── Lumen fill (dark blue) ──
        draw.rectangle(
            [pts_top[0][0], pts_top[0][1] + 2, pts_bot[-1][0], pts_bot[-1][1] - 2],
            fill=(10, 12, 30))

        # ── Endothelial cells (wall tiles) ──
        endo_pos = frame.get("endo_positions", [])
        selexpr = frame.get("endo_selectin_expr", [])
        junc = frame.get("endo_junction_integrity", [])
        for i, pos in enumerate(endo_pos):
            px, py = sim_to_px(pos[0], pos[1])
            se = selexpr[i] if i < len(selexpr) else 0.0
            ji = junc[i] if i < len(junc) else 1.0
            # Draw as small rectangles on vessel wall
            c = (int(160 + 80 * se), int(80 - 40 * se), int(100 - 60 * se))
            hw = int(4 + 3 * se)
            # Clamp to near vessel walls
            if pos[1] < 0:
                wall_y = int(cy + R * sy)
            else:
                wall_y = int(cy - R * sy)
            draw.rectangle([px - hw, wall_y - 2, px + hw, wall_y + 2], fill=c)
            # Selectin markers (yellow dots above endothelial cell)
            if se > 0.3:
                for si in range(int(se * 3)):
                    sy_off = 4 + si * 3
                    draw.ellipse([px - 1, wall_y - sy_off - 1,
                                  px + 1, wall_y - sy_off + 1],
                                 fill=(255, 220, 50))

        # ── RBCs (red ellipses) ──
        rbc_pos = frame.get("rbc_positions", [])
        for pos in rbc_pos:
            px, py = sim_to_px(pos[0], pos[1])
            # Only draw if inside vessel
            if pts_top[0][1] < py < pts_bot[0][1]:
                draw.ellipse([px - 4, py - 2, px + 4, py + 2],
                             fill=(180, 30, 30), outline=(120, 20, 20))

        # ── Leukocytes (colored circles by state) ──
        leuko_pos = frame.get("leukocyte_positions", [])
        leuko_states = frame.get("leukocyte_states", [])
        leuko_radii = frame.get("leukocyte_radii", [])
        integrin_act = frame.get("integrin_activation", [])
        for i, pos in enumerate(leuko_pos):
            px, py = sim_to_px(pos[0], pos[1])
            state = leuko_states[i] if i < len(leuko_states) else 0
            r = leuko_radii[i] if i < len(leuko_radii) else 6.0
            pr = max(3, int(r * sx * 0.4))
            color = self.STATE_COLORS.get(state, (200, 200, 200))
            # Draw cell body
            draw.ellipse([px - pr, py - pr, px + pr, py + pr],
                         fill=color, outline=(255, 255, 255))
            # Nucleus hint (darker inner circle)
            nr = max(1, pr // 2)
            nc = tuple(max(0, c - 60) for c in color)
            draw.ellipse([px - nr, py - nr, px + nr, py + nr], fill=nc)
            # Integrin activation indicator (bright ring)
            ia = integrin_act[i] if i < len(integrin_act) else 0.0
            if ia > 0.3:
                draw.ellipse([px - pr - 2, py - pr - 2, px + pr + 2, py + pr + 2],
                             outline=(0, 255, 255), width=1)

        # ── Bacteria (green in tissue space) ──
        bact_pos = frame.get("bacteria_positions", [])
        bact_alive = frame.get("bacteria_alive", [])
        bact_phago = frame.get("bacteria_phagocytosis", [])
        for i, pos in enumerate(bact_pos):
            px, py = sim_to_px(pos[0], pos[1])
            alive = bact_alive[i] if i < len(bact_alive) else True
            if alive:
                draw.ellipse([px - 3, py - 2, px + 3, py + 2],
                             fill=(40, 160, 40), outline=(200, 180, 60))
            else:
                draw.ellipse([px - 2, py - 1, px + 2, py + 1],
                             fill=(60, 60, 40), outline=(100, 80, 40))

        # ── Legend ──
        ly = 10
        draw.text((10, ly), "Cognisom Diapedesis", fill=(0, 160, 255), font=font_lg)
        ly += 22
        state_names = ["Flowing", "Tethered", "Rolling",
                       "Activating", "Arrested", "Transmigrating", "Extravasated"]
        for si, name in enumerate(state_names):
            c = self.STATE_COLORS.get(si, (200, 200, 200))
            draw.ellipse([10, ly, 18, ly + 8], fill=c)
            draw.text((22, ly - 1), name, fill=(180, 180, 200), font=font_sm)
            ly += 13

        # ── Metrics overlay (bottom-right) ──
        metrics = frame.get("metrics", {})
        t_val = frame.get("time", 0.0)
        step = frame.get("step", mgr.current_frame)
        info_lines = [
            f"Frame {mgr.current_frame}/{mgr.total_frames}",
            f"t = {t_val:.2f}s",
            f"Bacteria: {metrics.get('bacteria_alive', '?')}/{metrics.get('bacteria_total', '?')}",
        ]
        iy = H - 14 * len(info_lines) - 8
        for line in info_lines:
            draw.text((W - 200, iy), line, fill=(160, 170, 190), font=font)
            iy += 14

        # ── Flow arrow ──
        arrow_y = int(cy)
        draw.line([(margin_x, arrow_y), (W - margin_x, arrow_y)],
                  fill=(40, 50, 80), width=1)
        # Arrowhead
        ax = W - margin_x - 8
        draw.polygon([(ax, arrow_y - 4), (ax + 8, arrow_y), (ax, arrow_y + 4)],
                     fill=(60, 70, 100))
        draw.text((W - margin_x - 60, arrow_y - 14), "flow",
                  fill=(60, 80, 120), font=font_sm)

        # Encode
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        self._buffer = buf.getvalue()
        self._last_capture_time = time.time()
        return True

    def get_placeholder_jpeg(self) -> bytes:
        """Generate a placeholder image when no data is loaded."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new("RGB", (self._width, self._height), (10, 10, 30))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            draw.text((self._width // 2 - 140, self._height // 2 - 20),
                      "Cognisom Kit Server", fill=(0, 160, 255), font=font)
            draw.text((self._width // 2 - 120, self._height // 2 + 20),
                      "Waiting for data...", fill=(100, 100, 140), font=font)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        except Exception:
            return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


class _StreamHandler(BaseHTTPRequestHandler):
    """HTTP handler for Kit streaming endpoints."""

    # Class-level references (set by StreamingServer)
    diapedesis_manager = None
    viewport_capture = None

    def log_message(self, format, *args):
        """Suppress default logging — use carb instead."""
        pass

    def _send_json(self, data: Dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_jpeg(self, jpeg_bytes: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        self.wfile.write(jpeg_bytes)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/status":
            mgr = self.diapedesis_manager
            vc = self.viewport_capture
            self._send_json({
                "status": "ok",
                "kit": "running",
                "extension": "cognisom.sim-2.0.0",
                "renderer": "rtx" if (vc and vc._rtx_active) else "pil",
                "diapedesis_loaded": mgr is not None and mgr.total_frames > 0,
                "frames": mgr.total_frames if mgr else 0,
                "current_frame": mgr.current_frame if mgr else 0,
                "playing": mgr.is_playing if mgr else False,
            })

        elif path == "/frame.jpg":
            vc = self.viewport_capture
            if vc and vc.jpeg_bytes:
                self._send_jpeg(vc.jpeg_bytes)
            elif vc:
                self._send_jpeg(vc.get_placeholder_jpeg())
            else:
                self._send_json({"error": "viewport capture not available"}, 503)

        elif path == "/stream":
            # MJPEG stream (multipart/x-mixed-replace)
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            vc = self.viewport_capture
            try:
                while True:
                    if vc and vc.jpeg_bytes:
                        frame_data = vc.jpeg_bytes
                    elif vc:
                        frame_data = vc.get_placeholder_jpeg()
                    else:
                        break

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame_data)}\r\n\r\n".encode())
                    self.wfile.write(frame_data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(1.0 / 30.0)  # 30 fps cap
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif path == "/streaming/client":
            # Serve a simple HTML viewer page
            self._serve_viewer_html()

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)
        mgr = self.diapedesis_manager

        if not mgr:
            self._send_json({"error": "diapedesis manager not available"}, 503)
            return

        try:
            if path == "/cognisom/diapedesis":
                # Load frames, queue scene build for Kit main thread
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                data = json.loads(body)
                frames = data.get("frames", [])
                mgr.load_frames(frames, fps=data.get("fps", 30))
                mgr.request_build_scene()
                self._send_json({
                    "message": f"Loaded {len(frames)} frames, scene build queued",
                    "total_frames": mgr.total_frames,
                    "scene_built": False,
                })

            elif path == "/cognisom/diapedesis/play":
                mgr.play()
                self._send_json({"status": "playing"})

            elif path == "/cognisom/diapedesis/pause":
                if mgr.is_paused:
                    mgr.resume()
                    self._send_json({"status": "resumed"})
                else:
                    mgr.pause()
                    self._send_json({"status": "paused"})

            elif path == "/cognisom/diapedesis/stop":
                mgr.stop()
                self._send_json({"status": "stopped"})

            elif path == "/cognisom/diapedesis/seek":
                frame = int(params.get("frame", [0])[0])
                mgr.seek(frame)
                self._send_json({"status": "seeked", "frame": mgr.current_frame})

            elif path == "/cognisom/diapedesis/preset":
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len) if content_len else b"{}"
                data = json.loads(body) if body else {}
                preset = data.get("preset", "inflammation")
                duration = data.get("duration", 60)
                success = mgr.load_preset(preset, duration=duration)
                if success:
                    # Queue scene build for Kit main thread
                    mgr.request_build_scene()
                    self._send_json({
                        "message": f"Preset '{preset}' loaded, scene build queued",
                        "total_frames": mgr.total_frames,
                        "scene_built": False,
                    })
                else:
                    self._send_json({"error": f"Failed to load preset '{preset}'"}, 400)

            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as e:
            carb.log_warn(f"[streaming] POST {path} error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_viewer_html(self):
        """Serve a standalone HTML viewer with MJPEG streaming + controls.

        This viewer uses MJPEG for video (works everywhere) plus the REST
        API for simulation control. For the WebRTC viewer that uses
        NVIDIA's streaming library, build and serve webrtc-viewer/.
        """
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Cognisom RTX Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #040412; overflow: hidden; font: 13px monospace; color: #ccc; }
        #viewer { width: 100vw; height: calc(100vh - 44px); object-fit: contain; background: #040412; }
        #badge {
            position: fixed; top: 8px; right: 8px;
            background: rgba(180, 140, 0, 0.85); color: white;
            padding: 4px 12px; border-radius: 4px; font: 13px monospace; z-index: 10;
        }
        #info {
            position: fixed; top: 8px; left: 8px;
            background: rgba(4, 4, 18, 0.85); color: #aab;
            padding: 8px 14px; border-radius: 6px; font: 11px monospace;
            z-index: 10; line-height: 1.6; border: 1px solid rgba(100,100,150,0.15);
        }
        #controls {
            position: fixed; bottom: 0; left: 0; right: 0;
            background: rgba(4, 4, 18, 0.92); padding: 8px 16px;
            display: flex; align-items: center; gap: 12px;
            z-index: 10; border-top: 1px solid rgba(100,100,150,0.2);
        }
        button {
            padding: 5px 16px; cursor: pointer; background: #2a2a4a;
            color: #ccc; border: 1px solid #555; border-radius: 4px; font: 13px monospace;
        }
        button:hover { background: #3a3a6a; }
        #status { flex: 1; text-align: right; color: #778; }
    </style>
</head>
<body>
    <div id="badge">MJPEG</div>
    <div id="info"><b style="color:#00a0ff">Cognisom Diapedesis</b><br><span id="sim-info">Loading...</span></div>
    <img id="viewer" />
    <div id="controls">
        <button onclick="api('cognisom/diapedesis/play')">Play</button>
        <button onclick="api('cognisom/diapedesis/pause')">Pause</button>
        <button onclick="api('cognisom/diapedesis/stop')">Stop</button>
        <button onclick="loadPreset()">Load Preset</button>
        <span id="status">Connecting...</span>
    </div>
    <script>
        const base = window.location.pathname.replace(/\\/streaming\\/client.*$/, '');
        document.getElementById('viewer').src = base + '/stream';
        function api(path) { return fetch(base + '/' + path, {method:'POST'}); }
        function loadPreset() {
            fetch(base + '/cognisom/diapedesis/preset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({preset: 'inflammation', duration: 120}),
            }).then(r => r.json()).then(d => {
                document.getElementById('sim-info').textContent = d.message || 'Loaded';
            }).catch(() => {});
        }
        setInterval(async () => {
            try {
                const r = await fetch(base + '/status');
                const d = await r.json();
                const parts = [];
                if (d.diapedesis_loaded) parts.push(`Frame ${d.current_frame}/${d.frames}`);
                parts.push(d.playing ? 'Playing' : 'Idle');
                parts.push(`Renderer: ${d.renderer}`);
                document.getElementById('status').textContent = parts.join(' | ');
                const info = d.diapedesis_loaded
                    ? `Loaded: ${d.frames} frames | ${d.playing ? 'Playing' : 'Idle'}`
                    : 'No simulation loaded';
                document.getElementById('sim-info').textContent = info;
            } catch(e) { document.getElementById('status').textContent = 'Disconnected'; }
        }, 800);
    </script>
</body>
</html>"""
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server — each request in its own thread."""
    daemon_threads = True
    allow_reuse_address = True


class StreamingServer:
    """HTTP server for Kit viewport streaming."""

    def __init__(self, diapedesis_manager, port: int = 8211):
        self._port = port
        self._server: Optional[_ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._viewport_capture = _ViewportCapture()
        self._viewport_capture._diapedesis_mgr = diapedesis_manager
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False

        # Set class-level references for the handler
        _StreamHandler.diapedesis_manager = diapedesis_manager
        _StreamHandler.viewport_capture = self._viewport_capture

    def start(self):
        """Start the HTTP server in a daemon thread."""
        if self._running:
            return

        try:
            self._server = _ThreadingHTTPServer(
                ("0.0.0.0", self._port), _StreamHandler)
            self._server.timeout = 1.0
            self._running = True

            # Server thread
            self._thread = threading.Thread(
                target=self._serve_forever, daemon=True)
            self._thread.start()

            # Viewport capture thread (captures frames at ~30fps)
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True)
            self._capture_thread.start()

            carb.log_info(
                f"[streaming] HTTP streaming server started on port {self._port}")

        except Exception as e:
            carb.log_error(f"[streaming] Failed to start server: {e}")
            self._running = False

    def _serve_forever(self):
        """Run the HTTP server."""
        while self._running:
            self._server.handle_request()

    def _capture_loop(self):
        """Continuously capture viewport frames for MJPEG streaming.

        IMPORTANT: This thread must NOT drive playback (mgr.update) because
        the Kit main thread (_on_update in extension.py) already drives
        playback AND modifies the USD stage. Having two threads modify the
        USD stage concurrently causes race conditions and hangs.

        This loop only captures frames for MJPEG clients.
        """
        interval = 1.0 / 30.0
        while self._running:
            try:
                vc = self._viewport_capture
                # Only capture — playback is driven by Kit main thread
                vc.capture()
            except Exception:
                pass
            time.sleep(interval)

    def set_viewport(self, viewport_api):
        """Set viewport API reference for RTX frame capture."""
        self._viewport_capture._viewport_api = viewport_api
        carb.log_info("[streaming] Viewport API set for RTX capture")

    def set_annotator(self, rgb_annotator):
        """Set replicator annotator for render product capture."""
        self._viewport_capture._rgb_annotator = rgb_annotator
        carb.log_info("[streaming] Replicator annotator set for capture")

    def shutdown(self):
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.server_close()
        carb.log_info("[streaming] HTTP streaming server stopped")
