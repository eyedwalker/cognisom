"""
Cognisom Simulation Extension for Omniverse Kit
================================================

This extension provides real-time biological cell simulation
and diapedesis (leukocyte extravasation) visualization
in NVIDIA Omniverse applications (Create, Isaac Sim, etc.).

Features:
- Real-time tumor microenvironment simulation
- 7-step diapedesis cascade with RTX-quality rendering
- Cell physics using PhysX
- Live parameter editing
- Frame-based playback with timeline scrubbing
- Data streaming from Cognisom backend
"""

from .extension import CognisomSimExtension

__all__ = ["CognisomSimExtension"]
