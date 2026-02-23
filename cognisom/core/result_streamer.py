"""WebSocket result streamer for tissue-scale simulation.

Runs a lightweight WebSocket server on the simulation instance (Lambda Labs
or AWS GPU) and streams simulation snapshots to the Streamlit dashboard.

Snapshots are serialized with msgpack (fast, compact) or JSON (fallback).

Usage on the simulation instance::

    streamer = ResultStreamer(host="0.0.0.0", port=8600)
    streamer.start()
    # ... during simulation ...
    streamer.push_snapshot(snapshot_dict)
    # ... when done ...
    streamer.stop()

Usage on the dashboard (client)::

    client = SnapshotClient("ws://lambda-ip:8600")
    client.connect()
    for snapshot in client.receive_snapshots():
        update_dashboard(snapshot)
    client.close()

"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Try msgpack for efficient binary serialization
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


def _serialize_snapshot(snapshot: Dict) -> bytes:
    """Serialize a snapshot dict to bytes.

    Handles numpy arrays by converting to lists.
    Prefers msgpack, falls back to JSON.
    """
    # Convert numpy arrays to serializable form
    cleaned = _numpy_to_serializable(snapshot)

    if HAS_MSGPACK:
        return msgpack.packb(cleaned, use_bin_type=True)
    else:
        return json.dumps(cleaned).encode("utf-8")


def _deserialize_snapshot(data: bytes) -> Dict:
    """Deserialize bytes back to a snapshot dict."""
    if HAS_MSGPACK:
        return msgpack.unpackb(data, raw=False)
    else:
        return json.loads(data.decode("utf-8"))


def _numpy_to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON/msgpack serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_serializable(v) for v in obj]
    return obj


class ResultStreamer:
    """WebSocket server that streams simulation snapshots.

    Runs in a background thread. Push snapshots from the simulation
    loop; connected clients receive them in near-real-time.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8600,
        max_queue_size: int = 50,
    ):
        self._host = host
        self._port = port
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._clients: List[Any] = []
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._total_pushed = 0
        self._total_sent = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    @property
    def stats(self) -> Dict:
        return {
            "running": self._running,
            "n_clients": len(self._clients),
            "total_pushed": self._total_pushed,
            "total_sent": self._total_sent,
            "queue_size": self._queue.qsize(),
        }

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            log.warning("ResultStreamer already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            name="ResultStreamer",
            daemon=True,
        )
        self._thread.start()
        log.info("ResultStreamer started on ws://%s:%d", self._host, self._port)

    def stop(self) -> None:
        """Stop the server and disconnect all clients."""
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        log.info("ResultStreamer stopped (pushed=%d, sent=%d)",
                 self._total_pushed, self._total_sent)

    def push_snapshot(self, snapshot: Dict) -> None:
        """Push a snapshot to all connected clients.

        Non-blocking: drops oldest if queue is full.

        Args:
            snapshot: Dict with simulation state (positions, fields, metrics).
        """
        self._total_pushed += 1

        if not self._running:
            return

        try:
            data = _serialize_snapshot(snapshot)
        except Exception as e:
            log.warning("Failed to serialize snapshot: %s", e)
            return

        # Drop oldest if full
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass

        self._queue.put_nowait(data)

    def _run_server(self) -> None:
        """Run the asyncio WebSocket server (runs in background thread)."""
        try:
            import websockets
            import websockets.server
        except ImportError:
            log.error(
                "websockets package not installed. "
                "Install with: pip install websockets"
            )
            self._running = False
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def handler(websocket):
            self._clients.append(websocket)
            remote = websocket.remote_address
            log.info("Client connected: %s", remote)
            try:
                # Keep connection alive, send snapshots from queue
                while self._running:
                    try:
                        data = self._queue.get(timeout=0.1)
                        await websocket.send(data)
                        self._total_sent += 1
                    except queue.Empty:
                        # Send heartbeat ping
                        try:
                            await asyncio.wait_for(
                                websocket.ping(), timeout=5.0,
                            )
                        except Exception:
                            break
                    except Exception as e:
                        log.warning("Send error to %s: %s", remote, e)
                        break
            finally:
                self._clients.remove(websocket)
                log.info("Client disconnected: %s", remote)

        async def serve():
            async with websockets.server.serve(
                handler, self._host, self._port,
                max_size=50 * 1024 * 1024,  # 50 MB max message
            ):
                while self._running:
                    await asyncio.sleep(0.5)

        try:
            self._loop.run_until_complete(serve())
        except Exception as e:
            if self._running:
                log.error("WebSocket server error: %s", e)
        finally:
            self._loop.close()
            self._running = False


class SnapshotClient:
    """WebSocket client for receiving simulation snapshots.

    Used by the Streamlit dashboard to receive live updates
    from a remote simulation instance.
    """

    def __init__(self, url: str = "ws://localhost:8600"):
        self._url = url
        self._ws = None
        self._connected = False
        self._last_snapshot: Optional[Dict] = None
        self._receive_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_snapshot(self) -> Optional[Dict]:
        return self._last_snapshot

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to the simulation's WebSocket server.

        Returns True if connected successfully.
        """
        try:
            import websockets.sync.client
            self._ws = websockets.sync.client.connect(
                self._url,
                open_timeout=timeout,
                max_size=50 * 1024 * 1024,
            )
            self._connected = True
            log.info("Connected to %s", self._url)
            return True
        except Exception as e:
            log.warning("Failed to connect to %s: %s", self._url, e)
            self._connected = False
            return False

    def receive_snapshot(self, timeout: float = 5.0) -> Optional[Dict]:
        """Receive a single snapshot (blocking with timeout).

        Returns None if no snapshot available or connection lost.
        """
        if not self._connected or self._ws is None:
            return None

        try:
            data = self._ws.recv(timeout=timeout)
            if isinstance(data, str):
                data = data.encode("utf-8")
            snapshot = _deserialize_snapshot(data)
            self._last_snapshot = snapshot
            self._receive_count += 1
            return snapshot
        except TimeoutError:
            return None
        except Exception as e:
            log.warning("Receive error: %s", e)
            self._connected = False
            return None

    def receive_snapshots(self) -> Iterator[Dict]:
        """Generator that yields snapshots until disconnected."""
        while self._connected:
            snapshot = self.receive_snapshot()
            if snapshot is not None:
                yield snapshot

    def close(self) -> None:
        """Close the connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        self._connected = False
        log.info("SnapshotClient closed (%d snapshots received)",
                 self._receive_count)
