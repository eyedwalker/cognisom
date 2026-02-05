"""
Cognisom Admin Dashboard
========================

Streamlit-based platform UI for Cognisom.

Extensibility:
    New dashboard components can be registered at runtime:

        from cognisom.dashboard import register_component, ComponentType

        @register_component("my_panel", component_type=ComponentType.PANEL)
        def render_my_panel(state):
            import streamlit as st
            st.write("My custom panel")
"""

from .components import (
    # Types
    ComponentType,
    ComponentCategory,
    ComponentInfo,
    # Registry
    component_registry,
    register_component,
    get_component,
    get_component_info,
    list_components,
    list_pages,
    list_panels,
    render_component,
    render_components_by_type,
    discover_plugins,
)

__all__ = [
    # Types
    "ComponentType",
    "ComponentCategory",
    "ComponentInfo",
    # Registry
    "component_registry",
    "register_component",
    "get_component",
    "get_component_info",
    "list_components",
    "list_pages",
    "list_panels",
    "render_component",
    "render_components_by_type",
    "discover_plugins",
]
