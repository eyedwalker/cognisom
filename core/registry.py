"""
Plugin Registry System
======================

Generic, extensible registry pattern for dynamically registering and
discovering components at runtime. This is the foundation for Cognisom's
plugin architecture.

Registries:
    - EntityTypeRegistry: Biological entity types (genes, proteins, etc.)
    - PrimRegistry: Bio-USD prim types (BioCell, BioMolecule, etc.)
    - ModuleRegistry: Simulation modules (auto-discovered)
    - PhysicsModelRegistry: GPU physics backends
    - ComponentRegistry: Dashboard UI components

Usage::

    from cognisom.core.registry import Registry

    # Create a registry
    entity_registry = Registry("entity")

    # Register via decorator
    @entity_registry.register("virus")
    class VirusEntity:
        ...

    # Register via function
    entity_registry.register_class("bacterium", BacteriumEntity)

    # Lookup
    cls = entity_registry.get("virus")
    instance = entity_registry.create("virus", **kwargs)

    # Auto-discovery from entry points
    entity_registry.discover_plugins("cognisom.entities")

Phase 0 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import importlib
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

log = logging.getLogger(__name__)

# Type variable for registered classes
T = TypeVar("T")


@dataclass
class RegistryEntry(Generic[T]):
    """Metadata for a registered component."""

    name: str
    cls: Type[T]
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    deprecated_message: str = ""
    schema_version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.description and self.cls.__doc__:
            self.description = self.cls.__doc__.strip().split("\n")[0]


class RegistryError(Exception):
    """Base exception for registry operations."""
    pass


class DuplicateRegistrationError(RegistryError):
    """Raised when attempting to register a duplicate name."""
    pass


class NotFoundError(RegistryError):
    """Raised when a registered component is not found."""
    pass


class ValidationError(RegistryError):
    """Raised when a component fails validation."""
    pass


@runtime_checkable
class Registrable(Protocol):
    """Protocol that registered classes should implement for full integration."""

    @classmethod
    def registry_name(cls) -> str:
        """Return the canonical name for registry lookup."""
        ...

    @classmethod
    def registry_version(cls) -> str:
        """Return the version string."""
        ...


class Registry(Generic[T]):
    """
    Generic registry for dynamically registering and looking up components.

    Thread-safe and supports:
    - Decorator-based registration
    - Programmatic registration
    - Entry-point based plugin discovery
    - Validation hooks
    - Deprecation warnings
    - Version tracking

    Parameters
    ----------
    name : str
        Name of this registry (for logging)
    base_class : Type[T], optional
        If provided, all registered classes must be subclasses of this
    validators : List[Callable], optional
        Functions to validate classes before registration
    allow_override : bool
        Whether to allow re-registering the same name (default: False)

    Examples
    --------
    >>> registry = Registry[BioEntity]("entity", base_class=BioEntity)
    >>>
    >>> @registry.register("virus")
    ... class VirusEntity(BioEntity):
    ...     pass
    >>>
    >>> virus_cls = registry.get("virus")
    >>> instance = registry.create("virus", name="HIV-1")
    """

    def __init__(
        self,
        name: str,
        base_class: Optional[Type[T]] = None,
        validators: Optional[List[Callable[[Type[T]], bool]]] = None,
        allow_override: bool = False,
    ):
        self.name = name
        self.base_class = base_class
        self.validators = validators or []
        self.allow_override = allow_override
        self._entries: Dict[str, RegistryEntry[T]] = {}
        self._lock = threading.RLock()
        self._discovery_complete = False

        log.debug(f"Created registry: {name}")

    def register(
        self,
        name: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        tags: Optional[List[str]] = None,
        deprecated: bool = False,
        deprecated_message: str = "",
        schema_version: int = 1,
        **metadata: Any,
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a class.

        Parameters
        ----------
        name : str, optional
            Registration name (defaults to class name in snake_case)
        version : str
            Version string
        description : str
            Human-readable description
        author : str
            Author or maintainer
        tags : List[str]
            Searchable tags
        deprecated : bool
            Mark as deprecated
        deprecated_message : str
            Message to show for deprecated entries
        schema_version : int
            Schema version for serialization compatibility
        **metadata
            Additional metadata

        Returns
        -------
        Callable
            Decorator function

        Examples
        --------
        >>> @registry.register("my_entity", version="2.0.0", author="Lab")
        ... class MyEntity:
        ...     pass
        """
        def decorator(cls: Type[T]) -> Type[T]:
            reg_name = name or self._class_to_name(cls)
            self.register_class(
                reg_name,
                cls,
                version=version,
                description=description,
                author=author,
                tags=tags or [],
                deprecated=deprecated,
                deprecated_message=deprecated_message,
                schema_version=schema_version,
                **metadata,
            )
            return cls

        return decorator

    def register_class(
        self,
        name: str,
        cls: Type[T],
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        tags: Optional[List[str]] = None,
        deprecated: bool = False,
        deprecated_message: str = "",
        schema_version: int = 1,
        **metadata: Any,
    ) -> None:
        """
        Programmatically register a class.

        Parameters
        ----------
        name : str
            Registration name
        cls : Type[T]
            Class to register
        ... (same as register decorator)

        Raises
        ------
        DuplicateRegistrationError
            If name already registered and allow_override is False
        ValidationError
            If class fails validation
        """
        with self._lock:
            # Check for duplicates
            if name in self._entries and not self.allow_override:
                raise DuplicateRegistrationError(
                    f"'{name}' already registered in {self.name} registry. "
                    f"Existing: {self._entries[name].cls}, New: {cls}"
                )

            # Validate base class
            if self.base_class is not None:
                if not issubclass(cls, self.base_class):
                    raise ValidationError(
                        f"Class {cls} must be a subclass of {self.base_class}"
                    )

            # Run custom validators
            for validator in self.validators:
                if not validator(cls):
                    raise ValidationError(
                        f"Class {cls} failed validation by {validator.__name__}"
                    )

            entry = RegistryEntry(
                name=name,
                cls=cls,
                version=version,
                description=description,
                author=author,
                tags=tags or [],
                deprecated=deprecated,
                deprecated_message=deprecated_message,
                schema_version=schema_version,
                metadata=metadata,
            )

            self._entries[name] = entry

            if self.allow_override and name in self._entries:
                log.info(f"Overriding {self.name}/{name} with {cls}")
            else:
                log.debug(f"Registered {self.name}/{name}: {cls}")

    def unregister(self, name: str) -> bool:
        """
        Remove a registered class.

        Parameters
        ----------
        name : str
            Name to unregister

        Returns
        -------
        bool
            True if removed, False if not found
        """
        with self._lock:
            if name in self._entries:
                del self._entries[name]
                log.debug(f"Unregistered {self.name}/{name}")
                return True
            return False

    def get(self, name: str) -> Type[T]:
        """
        Get a registered class by name.

        Parameters
        ----------
        name : str
            Registration name

        Returns
        -------
        Type[T]
            The registered class

        Raises
        ------
        NotFoundError
            If name is not registered
        """
        entry = self.get_entry(name)

        if entry.deprecated:
            import warnings
            msg = f"'{name}' is deprecated in {self.name} registry."
            if entry.deprecated_message:
                msg += f" {entry.deprecated_message}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        return entry.cls

    def get_entry(self, name: str) -> RegistryEntry[T]:
        """
        Get the full registry entry for a name.

        Parameters
        ----------
        name : str
            Registration name

        Returns
        -------
        RegistryEntry[T]
            The entry with metadata

        Raises
        ------
        NotFoundError
            If name is not registered
        """
        with self._lock:
            if name not in self._entries:
                available = ", ".join(sorted(self._entries.keys())[:10])
                raise NotFoundError(
                    f"'{name}' not found in {self.name} registry. "
                    f"Available: {available}{'...' if len(self._entries) > 10 else ''}"
                )
            return self._entries[name]

    def create(self, entry_name: str, *args: Any, **kwargs: Any) -> T:
        """
        Create an instance of a registered class.

        Parameters
        ----------
        entry_name : str
            Registration name (the key used to register the class)
        *args, **kwargs
            Arguments passed to the constructor

        Returns
        -------
        T
            Instance of the registered class
        """
        cls = self.get(entry_name)
        return cls(*args, **kwargs)

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._entries

    def __len__(self) -> int:
        """Number of registered entries."""
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names."""
        return iter(self._entries.keys())

    def items(self) -> Iterator[tuple[str, RegistryEntry[T]]]:
        """Iterate over (name, entry) pairs."""
        return iter(self._entries.items())

    def list_names(self) -> List[str]:
        """Get list of all registered names."""
        with self._lock:
            return list(self._entries.keys())

    def list_entries(self) -> List[RegistryEntry[T]]:
        """Get list of all registry entries."""
        with self._lock:
            return list(self._entries.values())

    def filter_by_tag(self, tag: str) -> List[RegistryEntry[T]]:
        """Get entries that have a specific tag."""
        with self._lock:
            return [e for e in self._entries.values() if tag in e.tags]

    def discover_plugins(self, entry_point_group: str) -> int:
        """
        Discover and register plugins from entry points.

        This enables pip-installable plugins that auto-register.

        Parameters
        ----------
        entry_point_group : str
            Entry point group name (e.g., "cognisom.entities")

        Returns
        -------
        int
            Number of plugins discovered

        Examples
        --------
        In a plugin's pyproject.toml:

        [project.entry-points."cognisom.entities"]
        my_virus = "my_plugin.entities:VirusEntity"

        Then in cognisom:

        >>> registry.discover_plugins("cognisom.entities")
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points  # type: ignore

        count = 0

        try:
            # Python 3.10+ returns a SelectableGroups
            eps = entry_points(group=entry_point_group)
        except TypeError:
            # Python 3.9 returns a dict
            eps = entry_points().get(entry_point_group, [])

        for ep in eps:
            try:
                cls = ep.load()
                name = ep.name

                # Check if class implements Registrable protocol
                version = "1.0.0"
                if isinstance(cls, type) and hasattr(cls, "registry_version"):
                    try:
                        version = cls.registry_version()
                    except Exception:
                        pass
                if isinstance(cls, type) and hasattr(cls, "registry_name"):
                    try:
                        name = cls.registry_name()
                    except Exception:
                        pass

                self.register_class(name, cls, version=version, tags=["plugin"])
                count += 1
                log.info(f"Discovered plugin: {self.name}/{name} from {ep.value}")

            except Exception as e:
                log.warning(f"Failed to load plugin {ep.name}: {e}")

        self._discovery_complete = True
        return count

    def discover_modules(self, package_path: str, pattern: str = "*") -> int:
        """
        Discover and register classes from a package.

        Parameters
        ----------
        package_path : str
            Import path to package (e.g., "cognisom.modules")
        pattern : str
            Glob pattern for module names (default "*" matches all)

        Returns
        -------
        int
            Number of classes discovered
        """
        import importlib
        import pkgutil

        count = 0

        try:
            package = importlib.import_module(package_path)
        except ImportError as e:
            log.warning(f"Could not import package {package_path}: {e}")
            return 0

        if not hasattr(package, "__path__"):
            log.warning(f"{package_path} is not a package")
            return 0

        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
            if pattern != "*" and not self._match_pattern(modname, pattern):
                continue

            try:
                full_name = f"{package_path}.{modname}"
                module = importlib.import_module(full_name)

                # Look for classes that match our base class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if not isinstance(attr, type):
                        continue
                    if attr_name.startswith("_"):
                        continue
                    if self.base_class and not issubclass(attr, self.base_class):
                        continue
                    if attr is self.base_class:
                        continue

                    name = self._class_to_name(attr)
                    if name not in self._entries:
                        try:
                            self.register_class(name, attr, tags=["auto-discovered"])
                            count += 1
                        except (DuplicateRegistrationError, ValidationError):
                            pass

            except Exception as e:
                log.debug(f"Error loading module {modname}: {e}")

        return count

    def _class_to_name(self, cls: type) -> str:
        """Convert class name to registry name (snake_case)."""
        name = cls.__name__
        # Convert CamelCase to snake_case
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.lower())
        return "".join(result)

    def _match_pattern(self, name: str, pattern: str) -> bool:
        """Simple glob pattern matching."""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry state to dict."""
        return {
            "name": self.name,
            "count": len(self._entries),
            "entries": {
                name: {
                    "class": f"{e.cls.__module__}.{e.cls.__name__}",
                    "version": e.version,
                    "description": e.description,
                    "tags": e.tags,
                    "deprecated": e.deprecated,
                }
                for name, e in self._entries.items()
            },
        }

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, entries={len(self._entries)})"


class RegistryManager:
    """
    Central manager for all registries in the system.

    Provides a single access point for discovering and using
    all registered components.

    Examples
    --------
    >>> from cognisom.core.registry import registry_manager
    >>>
    >>> # Access registries
    >>> entity_cls = registry_manager.entities.get("virus")
    >>> prim_cls = registry_manager.prims.get("bio_cell")
    >>>
    >>> # List all registries
    >>> for name, reg in registry_manager.all():
    ...     print(f"{name}: {len(reg)} entries")
    """

    _instance: Optional["RegistryManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RegistryManager":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._registries: Dict[str, Registry] = {}
                    cls._instance._initialized = False
        return cls._instance

    def add_registry(self, name: str, registry: Registry) -> None:
        """Add a registry to the manager."""
        self._registries[name] = registry
        log.debug(f"Added registry to manager: {name}")

    def get_registry(self, name: str) -> Registry:
        """Get a registry by name."""
        if name not in self._registries:
            raise KeyError(f"Registry '{name}' not found. Available: {list(self._registries.keys())}")
        return self._registries[name]

    def __getattr__(self, name: str) -> Registry:
        """Allow attribute access to registries (e.g., manager.entities)."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.get_registry(name)
        except KeyError:
            raise AttributeError(f"No registry named '{name}'")

    def all(self) -> Iterator[tuple[str, Registry]]:
        """Iterate over all registries."""
        return iter(self._registries.items())

    def discover_all_plugins(self) -> Dict[str, int]:
        """
        Run plugin discovery on all registries.

        Returns
        -------
        Dict[str, int]
            Map of registry name -> plugins discovered
        """
        results = {}
        for name, registry in self._registries.items():
            # Use conventional entry point naming
            ep_group = f"cognisom.{name}"
            count = registry.discover_plugins(ep_group)
            results[name] = count
        return results

    def summary(self) -> str:
        """Get summary of all registries."""
        lines = ["Cognisom Registry Manager", "=" * 40]
        for name, reg in sorted(self._registries.items()):
            lines.append(f"  {name}: {len(reg)} entries")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all registries to dict."""
        return {
            name: reg.to_dict()
            for name, reg in self._registries.items()
        }


# Global registry manager instance
registry_manager = RegistryManager()


# Convenience function
def get_registry(name: str) -> Registry:
    """Get a registry from the global manager."""
    return registry_manager.get_registry(name)
