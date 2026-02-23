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
    """Captures the Kit viewport to a buffer."""

    def __init__(self):
        self._buffer: Optional[bytes] = None
        self._width = 1920
        self._height = 1080
        self._last_capture_time = 0.0

    @property
    def jpeg_bytes(self) -> Optional[bytes]:
        return self._buffer

    def capture(self) -> bool:
        """Capture current viewport frame."""
        if not VIEWPORT_AVAILABLE:
            return False

        try:
            viewport_api = vp_util.get_active_viewport()
            if not viewport_api:
                return False

            # Use capture_viewport_to_buffer for synchronous capture
            import asyncio
            import omni.kit.viewport.utility.capture as cap

            self._width = viewport_api.resolution[0]
            self._height = viewport_api.resolution[1]

            captured = [False]
            buffer_holder = [None]

            def on_capture_complete(buffer, buffer_size, width, height, fmt):
                if buffer:
                    buffer_holder[0] = bytes(buffer)
                    captured[0] = True

            cap.capture_viewport_to_buffer(
                viewport_api,
                on_capture_complete,
            )

            # Wait briefly for async capture
            for _ in range(10):
                if captured[0]:
                    break
                time.sleep(0.01)

            if buffer_holder[0]:
                self._buffer = _encode_jpeg(
                    buffer_holder[0], self._width, self._height)
                self._last_capture_time = time.time()
                return True

        except Exception as e:
            carb.log_warn(f"[streaming] Viewport capture failed: {e}")

        return False

    def get_placeholder_jpeg(self) -> bytes:
        """Generate a placeholder "waiting for RTX" image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new("RGB", (960, 540), (10, 10, 30))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            draw.text((300, 240), "Omniverse Kit RTX", fill=(0, 160, 255), font=font)
            draw.text((320, 280), "Waiting for scene...", fill=(100, 100, 140), font=font)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        except Exception:
            # Minimal 1x1 white JPEG
            return (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01"
                b"\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07"
                b"\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14"
                b"\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444"
                b"\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01"
                b"\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01"
                b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06"
                b"\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02"
                b"\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11"
                b"\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1"
                b"\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789"
                b":CDEFGHIJSTUVWXYZcdefghijstuvwxyz\xff\xda\x00\x08\x01\x01\x00"
                b"\x00?\x00\xfb\xd2\x8a(\x03\xff\xd9"
            )


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
            self._send_json({
                "status": "ok",
                "kit": "running",
                "extension": "cognisom.sim-2.0.0",
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
                # Load frames and build USD scene
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                data = json.loads(body)
                frames = data.get("frames", [])
                mgr.load_frames(frames, fps=data.get("fps", 30))
                built = mgr.build_scene()
                self._send_json({
                    "message": f"Loaded {len(frames)} frames, scene {'built' if built else 'build failed'}",
                    "total_frames": mgr.total_frames,
                    "scene_built": built,
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
                    built = mgr.build_scene()
                    self._send_json({
                        "message": f"Preset '{preset}' loaded, scene {'built' if built else 'pending'}",
                        "total_frames": mgr.total_frames,
                        "scene_built": built,
                    })
                else:
                    self._send_json({"error": f"Failed to load preset '{preset}'"}, 400)

            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as e:
            carb.log_warn(f"[streaming] POST {path} error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_viewer_html(self):
        """Serve a standalone HTML viewer for the MJPEG stream."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Cognisom RTX Viewer</title>
    <style>
        body { margin: 0; background: #0a0a1a; overflow: hidden; }
        #viewer { width: 100vw; height: 100vh; object-fit: contain; }
        #badge {
            position: fixed; top: 8px; right: 8px;
            background: rgba(0,180,0,0.85); color: white;
            padding: 4px 12px; border-radius: 4px;
            font: 13px monospace; z-index: 10;
        }
        #controls {
            position: fixed; bottom: 0; left: 0; right: 0;
            background: rgba(10,10,30,0.9); padding: 8px 16px;
            display: flex; align-items: center; gap: 12px;
            font: 13px monospace; color: #ccc; z-index: 10;
        }
        button {
            padding: 4px 14px; cursor: pointer; background: #2a2a4a;
            color: #ccc; border: 1px solid #555; border-radius: 4px;
        }
        button:hover { background: #3a3a6a; }
    </style>
</head>
<body>
    <div id="badge">RTX HD</div>
    <img id="viewer" src="/stream" />
    <div id="controls">
        <button onclick="fetch('/cognisom/diapedesis/play',{method:'POST'})">Play</button>
        <button onclick="fetch('/cognisom/diapedesis/pause',{method:'POST'})">Pause</button>
        <button onclick="fetch('/cognisom/diapedesis/stop',{method:'POST'})">Stop</button>
        <span id="status">Connecting...</span>
    </div>
    <script>
        setInterval(async () => {
            try {
                const r = await fetch('/status');
                const d = await r.json();
                document.getElementById('status').textContent =
                    `Frame ${d.current_frame}/${d.frames} | ${d.playing ? 'Playing' : 'Stopped'}`;
            } catch(e) {}
        }, 500);
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
        """Continuously capture viewport frames."""
        while self._running:
            try:
                self._viewport_capture.capture()
            except Exception:
                pass
            time.sleep(1.0 / 30.0)

    def shutdown(self):
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.server_close()
        carb.log_info("[streaming] HTTP streaming server stopped")
