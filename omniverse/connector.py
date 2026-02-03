"""
Omniverse Connector (Phase 8)
=============================

Connection manager for NVIDIA Omniverse and Isaac Sim.

Handles:
- Connection lifecycle (connect, disconnect, reconnect)
- Authentication and session management
- Live USD stage access
- Event subscription for scene changes

Requirements:
    - NVIDIA Omniverse Kit SDK
    - omni.client Python package (from Omniverse)

Usage::

    from cognisom.omniverse import OmniverseConnector, ConnectionStatus

    connector = OmniverseConnector()
    connector.connect("omniverse://localhost/cognisom")

    if connector.status == ConnectionStatus.CONNECTED:
        stage = connector.get_stage()
        # Work with USD stage
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from threading import Thread, Event

log = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """Connection status states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    AUTHENTICATED = "authenticated"


@dataclass
class ConnectionConfig:
    """Configuration for Omniverse connection."""
    url: str = "omniverse://localhost/cognisom"
    username: str = ""
    password: str = ""
    auto_reconnect: bool = True
    reconnect_interval: float = 5.0
    timeout: float = 30.0
    heartbeat_interval: float = 10.0
    stage_name: str = "cognisom_simulation.usd"


@dataclass
class ConnectionEvent:
    """Event from Omniverse connection."""
    event_type: str = ""      # "connected", "disconnected", "error", "stage_change"
    timestamp: float = 0.0
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class OmniverseConnector:
    """Connection manager for NVIDIA Omniverse.

    Provides connection lifecycle management and USD stage access
    for real-time simulation visualization.
    """

    def __init__(self, config: Optional[ConnectionConfig] = None) -> None:
        """Initialize the connector.

        Args:
            config: Connection configuration (defaults provided if None)
        """
        self._config = config or ConnectionConfig()
        self._status = ConnectionStatus.DISCONNECTED
        self._stage = None
        self._client = None
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._stop_event = Event()
        self._heartbeat_thread: Optional[Thread] = None
        self._last_heartbeat = 0.0
        self._connection_attempts = 0
        self._event_history: List[ConnectionEvent] = []

        # Omniverse Kit imports (lazy)
        self._omni_client = None
        self._omni_usd = None

    @property
    def status(self) -> ConnectionStatus:
        """Current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Whether connected to Omniverse."""
        return self._status in (ConnectionStatus.CONNECTED, ConnectionStatus.AUTHENTICATED)

    @property
    def config(self) -> ConnectionConfig:
        """Current configuration."""
        return self._config

    # ── Connection Lifecycle ────────────────────────────────────────────

    def connect(self, url: Optional[str] = None) -> bool:
        """Connect to Omniverse server.

        Args:
            url: Omniverse URL (uses config URL if not provided)

        Returns:
            True if connection successful
        """
        if url:
            self._config.url = url

        self._set_status(ConnectionStatus.CONNECTING)
        self._connection_attempts += 1

        try:
            # Try to import Omniverse modules
            if not self._init_omniverse():
                # Fall back to simulation mode
                log.info("Omniverse SDK not available, using simulation mode")
                return self._connect_simulated()

            # Initialize client
            self._omni_client.initialize()

            # Set up authentication if provided
            if self._config.username:
                self._omni_client.set_authentication_message_box_callback(
                    self._auth_callback
                )

            # Connect to server
            result = self._omni_client.stat(self._config.url)
            if result.status != self._omni_client.Result.OK:
                raise ConnectionError(f"Failed to connect: {result.status}")

            self._set_status(ConnectionStatus.CONNECTED)
            self._emit_event("connected", message=f"Connected to {self._config.url}")

            # Start heartbeat
            self._start_heartbeat()

            log.info("Connected to Omniverse: %s", self._config.url)
            return True

        except ImportError as e:
            log.warning("Omniverse SDK not available: %s", e)
            return self._connect_simulated()

        except Exception as e:
            self._set_status(ConnectionStatus.ERROR)
            self._emit_event("error", message=str(e))
            log.error("Connection failed: %s", e)

            if self._config.auto_reconnect:
                self._schedule_reconnect()

            return False

    def _connect_simulated(self) -> bool:
        """Connect in simulation mode (no real Omniverse)."""
        log.info("Running in simulation mode (no Omniverse connection)")
        self._set_status(ConnectionStatus.CONNECTED)
        self._emit_event("connected", message="Simulated connection", data={"simulated": True})
        self._start_heartbeat()
        return True

    def disconnect(self) -> None:
        """Disconnect from Omniverse."""
        self._stop_event.set()

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)

        if self._stage:
            self._close_stage()

        if self._omni_client:
            try:
                self._omni_client.shutdown()
            except Exception as e:
                log.warning("Error during shutdown: %s", e)

        self._set_status(ConnectionStatus.DISCONNECTED)
        self._emit_event("disconnected", message="Disconnected from Omniverse")
        log.info("Disconnected from Omniverse")

    def reconnect(self) -> bool:
        """Reconnect to Omniverse."""
        self._set_status(ConnectionStatus.RECONNECTING)
        self.disconnect()
        time.sleep(1.0)
        return self.connect()

    def _schedule_reconnect(self) -> None:
        """Schedule automatic reconnection."""
        def _reconnect():
            time.sleep(self._config.reconnect_interval)
            if self._status == ConnectionStatus.ERROR:
                log.info("Attempting automatic reconnection...")
                self.connect()

        thread = Thread(target=_reconnect, daemon=True)
        thread.start()

    # ── USD Stage Management ────────────────────────────────────────────

    def get_stage(self):
        """Get the current USD stage.

        Returns:
            USD stage object or None if not connected
        """
        if not self.is_connected:
            log.warning("Not connected to Omniverse")
            return None

        if self._stage is None:
            self._open_or_create_stage()

        return self._stage

    def _open_or_create_stage(self) -> None:
        """Open existing stage or create new one."""
        stage_url = f"{self._config.url}/{self._config.stage_name}"

        try:
            if self._omni_usd:
                # Try to open existing stage
                result = self._omni_client.stat(stage_url)
                if result.status == self._omni_client.Result.OK:
                    self._stage = self._omni_usd.open_stage(stage_url)
                    log.info("Opened existing stage: %s", stage_url)
                else:
                    # Create new stage
                    self._stage = self._omni_usd.create_stage(stage_url)
                    self._init_stage_defaults()
                    log.info("Created new stage: %s", stage_url)
            else:
                # Simulation mode - create mock stage
                self._stage = MockUSDStage(stage_url)
                log.info("Created mock stage for simulation mode")

        except Exception as e:
            log.error("Failed to open/create stage: %s", e)
            self._stage = MockUSDStage(stage_url)

    def _init_stage_defaults(self) -> None:
        """Initialize default stage structure."""
        if self._stage is None:
            return

        try:
            # Create default scene hierarchy
            root = self._stage.GetPseudoRoot()

            # Create hierarchy
            for path in ["/World", "/World/Cells", "/World/Environment",
                        "/World/Lights", "/World/Camera"]:
                self._stage.DefinePrim(path, "Xform")

            # Set up default lighting
            light_path = "/World/Lights/DomeLight"
            if self._omni_usd:
                light = self._stage.DefinePrim(light_path, "DomeLight")
                light.GetAttribute("intensity").Set(1000.0)

            self._stage.Save()
            log.info("Initialized stage with default structure")

        except Exception as e:
            log.warning("Failed to initialize stage defaults: %s", e)

    def _close_stage(self) -> None:
        """Close the current USD stage."""
        if self._stage:
            try:
                if hasattr(self._stage, "Save"):
                    self._stage.Save()
                self._stage = None
            except Exception as e:
                log.warning("Error closing stage: %s", e)

    def save_stage(self) -> bool:
        """Save the current stage."""
        if self._stage:
            try:
                self._stage.Save()
                log.info("Stage saved")
                return True
            except Exception as e:
                log.error("Failed to save stage: %s", e)
        return False

    # ── Event System ────────────────────────────────────────────────────

    def on(self, event_type: str, handler: Callable) -> None:
        """Register event handler.

        Args:
            event_type: Event type to handle
            handler: Callback function(event: ConnectionEvent)
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable) -> None:
        """Unregister event handler."""
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
            except ValueError:
                pass

    def _emit_event(self, event_type: str, message: str = "",
                   data: Optional[Dict] = None) -> None:
        """Emit an event to registered handlers."""
        event = ConnectionEvent(
            event_type=event_type,
            message=message,
            data=data or {}
        )
        self._event_history.append(event)

        # Keep history bounded
        if len(self._event_history) > 100:
            self._event_history = self._event_history[-50:]

        # Call handlers
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                log.error("Event handler error: %s", e)

    def _set_status(self, status: ConnectionStatus) -> None:
        """Update connection status."""
        old_status = self._status
        self._status = status
        if old_status != status:
            self._emit_event("status_change", message=f"{old_status} -> {status}",
                           data={"old": old_status, "new": status})

    # ── Heartbeat ───────────────────────────────────────────────────────

    def _start_heartbeat(self) -> None:
        """Start heartbeat thread."""
        self._stop_event.clear()
        self._heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        """Heartbeat loop to maintain connection."""
        while not self._stop_event.is_set():
            try:
                if self.is_connected and self._omni_client:
                    result = self._omni_client.stat(self._config.url)
                    if result.status != self._omni_client.Result.OK:
                        log.warning("Heartbeat failed, connection may be lost")
                        self._set_status(ConnectionStatus.ERROR)
                        if self._config.auto_reconnect:
                            self.reconnect()
                            return

                self._last_heartbeat = time.time()

            except Exception as e:
                log.warning("Heartbeat error: %s", e)

            self._stop_event.wait(self._config.heartbeat_interval)

    # ── Initialization ──────────────────────────────────────────────────

    def _init_omniverse(self) -> bool:
        """Initialize Omniverse SDK modules."""
        try:
            import omni.client as omni_client
            import omni.usd as omni_usd
            self._omni_client = omni_client
            self._omni_usd = omni_usd
            return True
        except ImportError:
            return False

    def _auth_callback(self, server: str, auth_handle) -> bool:
        """Handle authentication callback."""
        if self._config.username and self._config.password:
            auth_handle.set_authentication(
                self._config.username,
                self._config.password
            )
            return True
        return False

    # ── Status and Info ─────────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            "status": self._status.value,
            "url": self._config.url,
            "stage_name": self._config.stage_name,
            "is_connected": self.is_connected,
            "connection_attempts": self._connection_attempts,
            "last_heartbeat": self._last_heartbeat,
            "has_stage": self._stage is not None,
            "simulated": self._omni_client is None,
        }

    def get_event_history(self, limit: int = 20) -> List[ConnectionEvent]:
        """Get recent connection events."""
        return self._event_history[-limit:]


class MockUSDStage:
    """Mock USD stage for simulation mode without Omniverse."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._prims: Dict[str, MockPrim] = {}
        self._root = MockPrim("/")

        # Initialize default structure
        for path in ["/World", "/World/Cells", "/World/Environment"]:
            self.DefinePrim(path, "Xform")

    def GetPseudoRoot(self):
        return self._root

    def DefinePrim(self, path: str, prim_type: str = "Xform"):
        prim = MockPrim(path, prim_type)
        self._prims[path] = prim
        return prim

    def GetPrimAtPath(self, path: str):
        return self._prims.get(path)

    def RemovePrim(self, path: str) -> bool:
        if path in self._prims:
            del self._prims[path]
            return True
        return False

    def Save(self) -> None:
        log.debug("Mock stage save (no-op)")

    def GetRootLayer(self):
        return MockLayer()


class MockPrim:
    """Mock USD prim."""

    def __init__(self, path: str, prim_type: str = "Xform") -> None:
        self._path = path
        self._type = prim_type
        self._attributes: Dict[str, Any] = {}
        self._children: List[MockPrim] = []

    def GetPath(self) -> str:
        return self._path

    def GetTypeName(self) -> str:
        return self._type

    def GetAttribute(self, name: str):
        if name not in self._attributes:
            self._attributes[name] = MockAttribute(name)
        return self._attributes[name]

    def CreateAttribute(self, name: str, attr_type=None):
        attr = MockAttribute(name)
        self._attributes[name] = attr
        return attr

    def GetChildren(self):
        return self._children

    def IsValid(self) -> bool:
        return True


class MockAttribute:
    """Mock USD attribute."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._value = None

    def Get(self):
        return self._value

    def Set(self, value) -> None:
        self._value = value


class MockLayer:
    """Mock USD layer."""

    def GetIdentifier(self) -> str:
        return "mock_layer"

    def Export(self, path: str) -> bool:
        return True
