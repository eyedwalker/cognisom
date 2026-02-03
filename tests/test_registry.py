"""
Tests for the Registry System
=============================

Tests for the Phase 0 extensibility framework including:
- Core Registry class
- Entity type registry
- Bio-USD prim registry
- Module registry
- Physics model registry
- Dashboard component registry
- Plugin auto-discovery
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict


# ═══════════════════════════════════════════════════════════════════════
# Core Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRegistry:
    """Tests for the core Registry class."""

    def test_registry_creation(self):
        """Test basic registry creation."""
        from cognisom.core.registry import Registry

        registry = Registry("test")
        assert registry.name == "test"
        assert len(registry) == 0

    def test_register_class(self):
        """Test registering a class programmatically."""
        from cognisom.core.registry import Registry

        registry = Registry("test")

        class TestClass:
            pass

        registry.register_class("test_class", TestClass)

        assert "test_class" in registry
        assert registry.get("test_class") is TestClass

    def test_register_decorator(self):
        """Test registering a class with decorator."""
        from cognisom.core.registry import Registry

        registry = Registry("test")

        @registry.register("decorated_class", version="2.0.0")
        class DecoratedClass:
            pass

        assert "decorated_class" in registry
        entry = registry.get_entry("decorated_class")
        assert entry.version == "2.0.0"

    def test_create_instance(self):
        """Test creating instances from registry."""
        from cognisom.core.registry import Registry

        registry = Registry("test")

        @dataclass
        class Point:
            x: float = 0.0
            y: float = 0.0

        registry.register_class("point", Point)

        point = registry.create("point", x=10.0, y=20.0)
        assert point.x == 10.0
        assert point.y == 20.0

    def test_duplicate_registration_error(self):
        """Test that duplicate registration raises error."""
        from cognisom.core.registry import Registry, DuplicateRegistrationError

        registry = Registry("test", allow_override=False)

        class A:
            pass

        class B:
            pass

        registry.register_class("same_name", A)

        with pytest.raises(DuplicateRegistrationError):
            registry.register_class("same_name", B)

    def test_allow_override(self):
        """Test that override works when enabled."""
        from cognisom.core.registry import Registry

        registry = Registry("test", allow_override=True)

        class A:
            pass

        class B:
            pass

        registry.register_class("same_name", A)
        registry.register_class("same_name", B)

        assert registry.get("same_name") is B

    def test_not_found_error(self):
        """Test that NotFoundError is raised for missing entries."""
        from cognisom.core.registry import Registry, NotFoundError

        registry = Registry("test")

        with pytest.raises(NotFoundError):
            registry.get("nonexistent")

    def test_base_class_validation(self):
        """Test that base class validation works."""
        from cognisom.core.registry import Registry, ValidationError

        class BaseClass:
            pass

        class ValidChild(BaseClass):
            pass

        class InvalidClass:
            pass

        registry = Registry("test", base_class=BaseClass)

        # Should succeed
        registry.register_class("valid", ValidChild)

        # Should fail
        with pytest.raises(ValidationError):
            registry.register_class("invalid", InvalidClass)

    def test_list_names(self):
        """Test listing registered names."""
        from cognisom.core.registry import Registry

        registry = Registry("test")

        class A:
            pass

        class B:
            pass

        registry.register_class("a", A)
        registry.register_class("b", B)

        names = registry.list_names()
        assert "a" in names
        assert "b" in names

    def test_filter_by_tag(self):
        """Test filtering entries by tag."""
        from cognisom.core.registry import Registry

        registry = Registry("test")

        class A:
            pass

        class B:
            pass

        registry.register_class("a", A, tags=["category1"])
        registry.register_class("b", B, tags=["category2"])

        cat1 = registry.filter_by_tag("category1")
        assert len(cat1) == 1
        assert cat1[0].name == "a"

    def test_deprecation_warning(self):
        """Test that deprecated entries emit warnings."""
        from cognisom.core.registry import Registry
        import warnings

        registry = Registry("test")

        class OldClass:
            pass

        registry.register_class(
            "old_class",
            OldClass,
            deprecated=True,
            deprecated_message="Use new_class instead",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry.get("old_class")
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()


# ═══════════════════════════════════════════════════════════════════════
# Entity Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEntityRegistry:
    """Tests for the entity type registry."""

    def test_builtin_entities_registered(self):
        """Test that all built-in entities are registered."""
        from cognisom.library.models import entity_registry, list_entity_types

        types = list_entity_types()

        assert "gene" in types
        assert "protein" in types
        assert "drug" in types
        assert "metabolite" in types

    def test_get_entity_class(self):
        """Test retrieving entity classes."""
        from cognisom.library.models import get_entity_class, Gene

        cls = get_entity_class("gene")
        assert cls is Gene

    def test_create_entity(self):
        """Test creating entities via registry."""
        from cognisom.library.models import create_entity

        gene = create_entity("gene", name="TP53", symbol="TP53")
        assert gene.name == "TP53"
        assert gene.symbol == "TP53"

    def test_custom_entity_registration(self):
        """Test registering a custom entity type."""
        from cognisom.library.models import (
            entity_registry,
            BioEntity,
            register_entity,
        )

        @register_entity("test_pathogen", version="1.0.0")
        @dataclass
        class TestPathogen(BioEntity):
            pathogen_type: str = ""

        assert "test_pathogen" in entity_registry


# ═══════════════════════════════════════════════════════════════════════
# Bio-USD Prim Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPrimRegistry:
    """Tests for the Bio-USD prim registry."""

    def test_builtin_prims_registered(self):
        """Test that all built-in prims are registered."""
        from cognisom.biousd.schema import prim_registry, list_prim_types

        types = list_prim_types()

        assert "bio_cell" in types
        assert "bio_protein" in types
        assert "bio_exosome" in types

    def test_get_prim_class(self):
        """Test retrieving prim classes."""
        from cognisom.biousd.schema import get_prim_class, BioCell

        cls = get_prim_class("bio_cell")
        assert cls is BioCell

    def test_create_prim(self):
        """Test creating prims via registry."""
        from cognisom.biousd.schema import create_prim

        cell = create_prim("bio_cell", cell_id=123, position=(10.0, 20.0, 30.0))
        assert cell.cell_id == 123
        assert cell.position == (10.0, 20.0, 30.0)

    def test_api_registry(self):
        """Test that API schemas are registered."""
        from cognisom.biousd.schema import api_registry, list_api_schemas

        schemas = list_api_schemas()

        assert "bio_metabolic_api" in schemas
        assert "bio_immune_api" in schemas


# ═══════════════════════════════════════════════════════════════════════
# Module Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestModuleRegistry:
    """Tests for the simulation module registry."""

    def test_builtin_modules_registered(self):
        """Test that all built-in modules are registered."""
        from cognisom.modules import module_registry, list_modules

        modules = list_modules()

        assert "cellular" in modules
        assert "immune" in modules
        assert "spatial" in modules

    def test_get_module_class(self):
        """Test retrieving module classes."""
        from cognisom.modules import get_module_class, CellularModule

        cls = get_module_class("cellular")
        assert cls is CellularModule

    def test_create_module(self):
        """Test creating modules via registry."""
        from cognisom.modules import create_module

        module = create_module("cellular", config={"test": True})
        assert module.config.get("test") is True


# ═══════════════════════════════════════════════════════════════════════
# Physics Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPhysicsRegistry:
    """Tests for the GPU physics model registry."""

    def test_builtin_physics_registered(self):
        """Test that built-in physics models are registered."""
        from cognisom.gpu.physics_interface import (
            physics_registry,
            list_physics_models,
        )

        models = list_physics_models()

        assert "cupy_diffusion" in models
        assert "cupy_particle" in models
        assert "numpy_fallback" in models

    def test_get_physics_model(self):
        """Test retrieving physics model classes."""
        from cognisom.gpu.physics_interface import (
            get_physics_model,
            CuPyParticlePhysics,
        )

        cls = get_physics_model("cupy_particle")
        assert cls is CuPyParticlePhysics

    def test_filter_by_backend(self):
        """Test filtering physics by backend type."""
        from cognisom.gpu.physics_interface import (
            list_physics_by_backend,
            PhysicsBackendType,
        )

        cupy_models = list_physics_by_backend(PhysicsBackendType.CUPY)
        assert "cupy_diffusion" in cupy_models
        assert "cupy_particle" in cupy_models


# ═══════════════════════════════════════════════════════════════════════
# Dashboard Component Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestComponentRegistry:
    """Tests for the dashboard component registry."""

    def test_builtin_components_registered(self):
        """Test that built-in components are registered."""
        from cognisom.dashboard.components import (
            component_registry,
            list_components,
        )

        components = list_components()

        assert "simulation_status" in components
        assert "registry_status" in components

    def test_get_component(self):
        """Test retrieving component functions."""
        from cognisom.dashboard.components import get_component

        func = get_component("simulation_status")
        assert callable(func)

    def test_get_component_info(self):
        """Test retrieving component metadata."""
        from cognisom.dashboard.components import (
            get_component_info,
            ComponentType,
        )

        info = get_component_info("simulation_status")
        assert info.component_type == ComponentType.WIDGET
        assert info.icon == "⚡"


# ═══════════════════════════════════════════════════════════════════════
# Registry Manager Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRegistryManager:
    """Tests for the central registry manager."""

    def test_singleton_pattern(self):
        """Test that registry manager is a singleton."""
        from cognisom.core.registry import RegistryManager

        manager1 = RegistryManager()
        manager2 = RegistryManager()
        assert manager1 is manager2

    def test_get_registry(self):
        """Test getting registries from manager."""
        from cognisom.core.registry import registry_manager

        entities = registry_manager.get_registry("entities")
        assert entities is not None

        prims = registry_manager.get_registry("prims")
        assert prims is not None

    def test_attribute_access(self):
        """Test attribute-style registry access."""
        from cognisom.core.registry import registry_manager

        entities = registry_manager.entities
        assert entities is not None

    def test_all_registries(self):
        """Test iterating over all registries."""
        from cognisom.core.registry import registry_manager

        registries = list(registry_manager.all())
        names = [name for name, _ in registries]

        assert "entities" in names
        assert "prims" in names
        assert "modules" in names

    def test_summary(self):
        """Test getting summary of all registries."""
        from cognisom.core.registry import registry_manager

        summary = registry_manager.summary()
        assert "entities" in summary
        assert "prims" in summary


# ═══════════════════════════════════════════════════════════════════════
# Virus Plugin Tests
# ═══════════════════════════════════════════════════════════════════════

class TestVirusPlugin:
    """Tests for the example virus plugin."""

    def test_plugin_loads(self):
        """Test that the virus plugin can be loaded."""
        import cognisom.plugins.examples.virus_plugin

        # Should not raise any errors
        assert True

    def test_virus_entity_registered(self):
        """Test that VirusEntity is registered."""
        import cognisom.plugins.examples.virus_plugin
        from cognisom.library.models import entity_registry

        assert "virus" in entity_registry

    def test_virus_prim_registered(self):
        """Test that BioVirusParticle is registered."""
        import cognisom.plugins.examples.virus_plugin
        from cognisom.biousd.schema import prim_registry

        assert "bio_virus_particle" in prim_registry

    def test_virus_module_registered(self):
        """Test that VirusModule is registered."""
        import cognisom.plugins.examples.virus_plugin
        from cognisom.modules import module_registry

        assert "virus" in module_registry

    def test_virus_components_registered(self):
        """Test that virus dashboard components are registered."""
        import cognisom.plugins.examples.virus_plugin
        from cognisom.dashboard.components import component_registry

        assert "virus_tracker" in component_registry
        assert "virus_seeder" in component_registry

    def test_create_virus_entity(self):
        """Test creating a virus entity."""
        import cognisom.plugins.examples.virus_plugin
        from cognisom.plugins.examples.virus_plugin import (
            VirusEntity,
            VirusFamily,
            VirusGenomeType,
        )

        virus = VirusEntity(
            name="SARS-CoV-2",
            virus_family=VirusFamily.CORONAVIRUS,
            genome_type=VirusGenomeType.RNA_POSITIVE,
            genome_length=29903,
            host_receptors=["ACE2"],
            envelope=True,
        )

        assert virus.name == "SARS-CoV-2"
        assert virus.genome_length == 29903

    def test_create_virus_prim(self):
        """Test creating a virus prim."""
        import cognisom.plugins.examples.virus_plugin
        from cognisom.plugins.examples.virus_plugin import BioVirusParticle

        prim = BioVirusParticle(
            virus_id=1,
            position=(50.0, 50.0, 25.0),
        )

        assert prim.virus_id == 1
        assert prim.position == (50.0, 50.0, 25.0)

    def test_virus_module_initialization(self):
        """Test initializing the virus module."""
        import cognisom.plugins.examples.virus_plugin
        from cognisom.plugins.examples.virus_plugin import VirusModule

        module = VirusModule(config={"diffusion_rate": 15.0})
        module.initialize()

        assert module.diffusion_rate == 15.0
        assert len(module.viruses) == 0

    def test_virus_module_seeding(self):
        """Test seeding viruses."""
        import cognisom.plugins.examples.virus_plugin
        from cognisom.plugins.examples.virus_plugin import VirusModule

        module = VirusModule()
        module.initialize()

        module.seed_viruses(100, (0, 0, 0, 100, 100, 50))

        state = module.get_state()
        assert state["active_viruses"] == 100
