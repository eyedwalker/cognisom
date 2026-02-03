"""
Cognisom Simulation Extension for Omniverse Kit
================================================

This extension provides real-time biological cell simulation
in NVIDIA Omniverse applications (Create, Isaac Sim, etc.).

Features:
- Real-time tumor microenvironment simulation
- Cell physics using PhysX
- Live parameter editing
- Data streaming from Cognisom backend
"""

from .extension import CognisomSimExtension

__all__ = ["CognisomSimExtension"]
