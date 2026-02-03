"""
Dashboard Component Registry
=============================

Registry for pluggable dashboard UI components. Enables plugins to add
new visualizations, panels, and widgets to the Streamlit dashboard
without modifying the core application.

Component Types:
    - Page: Full dashboard page (appears in sidebar)
    - Panel: Embedded panel within a page
    - Widget: Small reusable component
    - Visualization: Chart or 3D view

Usage::

    from cognisom.dashboard.components import register_component, ComponentType

    @register_component(
        "virus_tracker",
        component_type=ComponentType.PANEL,
        title="Virus Tracker",
        icon="🦠",
    )
    def render_virus_tracker(state):
        import streamlit as st
        st.write("Virus count:", state.get("virus_count", 0))

Phase 0 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from cognisom.core.registry import Registry, registry_manager

log = logging.getLogger(__name__)


class ComponentType(str, Enum):
    """Types of dashboard components."""
    PAGE = "page"             # Full page (sidebar entry)
    PANEL = "panel"           # Panel within a page
    WIDGET = "widget"         # Small reusable widget
    VISUALIZATION = "visualization"  # Chart/3D view
    SIDEBAR = "sidebar"       # Sidebar widget
    METRIC = "metric"         # Metric card


class ComponentCategory(str, Enum):
    """Categories for grouping components."""
    SIMULATION = "simulation"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    DATA = "data"
    ADMIN = "admin"
    CUSTOM = "custom"


@dataclass
class ComponentInfo:
    """
    Metadata for a registered UI component.
    """
    name: str
    title: str
    component_type: ComponentType
    category: ComponentCategory = ComponentCategory.CUSTOM
    icon: str = ""
    description: str = ""
    requires_auth: bool = True
    min_tier: str = "free"  # Subscription tier required
    position: int = 100  # Sort order (lower = earlier)


# ── Component Registry ───────────────────────────────────────────────

component_registry = Registry(
    name="components",
    base_class=None,  # Accept functions or classes
    allow_override=False,
)
registry_manager.add_registry("components", component_registry)


def register_component(
    name: str,
    component_type: ComponentType = ComponentType.WIDGET,
    title: str = "",
    category: ComponentCategory = ComponentCategory.CUSTOM,
    icon: str = "",
    description: str = "",
    requires_auth: bool = True,
    min_tier: str = "free",
    position: int = 100,
    version: str = "1.0.0",
    **metadata
):
    """
    Decorator to register a dashboard component.

    The decorated function should accept the current application state
    and render Streamlit elements.

    Parameters
    ----------
    name : str
        Component identifier (e.g., "virus_panel")
    component_type : ComponentType
        Type of component
    title : str
        Display title (defaults to name)
    category : ComponentCategory
        Category for grouping
    icon : str
        Emoji or icon for display
    description : str
        Human-readable description
    requires_auth : bool
        Whether authentication is required
    min_tier : str
        Minimum subscription tier
    position : int
        Sort order (lower numbers appear first)
    version : str
        Component version

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @register_component(
    ...     "virus_tracker",
    ...     component_type=ComponentType.PANEL,
    ...     title="Virus Tracker",
    ...     icon="🦠",
    ...     category=ComponentCategory.SIMULATION,
    ... )
    ... def render_virus_tracker(state):
    ...     import streamlit as st
    ...     st.subheader("🦠 Virus Tracker")
    ...     st.metric("Active Viruses", state.get("virus_count", 0))
    """
    def decorator(func: Callable):
        component_registry.register_class(
            name,
            func,
            version=version,
            component_type=component_type.value,
            title=title or name.replace("_", " ").title(),
            category=category.value,
            icon=icon,
            description=description or (func.__doc__ or "").strip().split("\n")[0],
            requires_auth=requires_auth,
            min_tier=min_tier,
            position=position,
            **metadata,
        )
        return func

    return decorator


def get_component(name: str) -> Callable:
    """
    Get a component render function by name.

    Parameters
    ----------
    name : str
        Component name

    Returns
    -------
    Callable
        The component render function
    """
    return component_registry.get(name)


def get_component_info(name: str) -> ComponentInfo:
    """
    Get component metadata by name.

    Parameters
    ----------
    name : str
        Component name

    Returns
    -------
    ComponentInfo
        Component metadata
    """
    entry = component_registry.get_entry(name)
    return ComponentInfo(
        name=name,
        title=entry.metadata.get("title", name),
        component_type=ComponentType(entry.metadata.get("component_type", "widget")),
        category=ComponentCategory(entry.metadata.get("category", "custom")),
        icon=entry.metadata.get("icon", ""),
        description=entry.description,
        requires_auth=entry.metadata.get("requires_auth", True),
        min_tier=entry.metadata.get("min_tier", "free"),
        position=entry.metadata.get("position", 100),
    )


def list_components(
    component_type: Optional[ComponentType] = None,
    category: Optional[ComponentCategory] = None,
) -> List[str]:
    """
    List registered components, optionally filtered.

    Parameters
    ----------
    component_type : ComponentType, optional
        Filter by component type
    category : ComponentCategory, optional
        Filter by category

    Returns
    -------
    List[str]
        Component names
    """
    names = []
    for name, entry in component_registry.items():
        if component_type and entry.metadata.get("component_type") != component_type.value:
            continue
        if category and entry.metadata.get("category") != category.value:
            continue
        names.append(name)

    # Sort by position
    def get_position(n):
        try:
            return component_registry.get_entry(n).metadata.get("position", 100)
        except Exception:
            return 100

    return sorted(names, key=get_position)


def list_pages() -> List[str]:
    """List all registered page components."""
    return list_components(component_type=ComponentType.PAGE)


def list_panels() -> List[str]:
    """List all registered panel components."""
    return list_components(component_type=ComponentType.PANEL)


def render_component(name: str, state: Dict[str, Any] = None) -> None:
    """
    Render a component by name.

    Parameters
    ----------
    name : str
        Component name
    state : Dict, optional
        Application state to pass to component
    """
    func = get_component(name)
    func(state or {})


def render_components_by_type(
    component_type: ComponentType,
    state: Dict[str, Any] = None,
) -> None:
    """
    Render all components of a given type.

    Parameters
    ----------
    component_type : ComponentType
        Type of components to render
    state : Dict, optional
        Application state
    """
    for name in list_components(component_type=component_type):
        try:
            render_component(name, state)
        except Exception as e:
            log.error(f"Error rendering component {name}: {e}")


def discover_plugins(entry_point_group: str = "cognisom.components") -> int:
    """
    Discover and register components from entry points.

    Parameters
    ----------
    entry_point_group : str
        Entry point group name

    Returns
    -------
    int
        Number of plugins discovered
    """
    return component_registry.discover_plugins(entry_point_group)


# ── Built-in Components ──────────────────────────────────────────────

@register_component(
    "simulation_status",
    component_type=ComponentType.WIDGET,
    title="Simulation Status",
    icon="⚡",
    category=ComponentCategory.SIMULATION,
    requires_auth=False,
    position=10,
)
def render_simulation_status(state: Dict[str, Any]):
    """Display current simulation status."""
    import streamlit as st

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cells", state.get("cell_count", 0))
    with col2:
        st.metric("Time (h)", f"{state.get('sim_time', 0):.1f}")
    with col3:
        status = "Running" if state.get("running", False) else "Stopped"
        st.metric("Status", status)


@register_component(
    "registry_status",
    component_type=ComponentType.WIDGET,
    title="Registry Status",
    icon="📦",
    category=ComponentCategory.ADMIN,
    min_tier="institution",
    position=90,
)
def render_registry_status(state: Dict[str, Any]):
    """Display status of all registries."""
    import streamlit as st

    st.subheader("📦 Registry Status")

    for name, registry in registry_manager.all():
        st.text(f"  {name}: {len(registry)} entries")


log.debug(f"Component registry initialized with {len(component_registry)} components")
