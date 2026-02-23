"""Tests for ParameterBridge and ScenarioBuilder."""

import pytest
from unittest.mock import MagicMock, patch

from cognisom.library.models import (
    Drug,
    Gene,
    ParameterSet,
    SimulationScenario,
)
from cognisom.workflow.parameter_bridge import (
    BASELINE_DEFAULTS,
    DRUG_CLASS_PARAM_MAP,
    DRUG_TARGET_PARAM_MAP,
    MODULE_PARAM_SCHEMA,
    ParameterBridge,
)
from cognisom.workflow.scenario_builder import ScenarioBuilder


# ── Fixtures ─────────────────────────────────────────────────────────

class MockStore:
    """Minimal EntityStore mock for testing."""

    def __init__(self):
        self._entities = {}

    def add_entity(self, entity):
        self._entities[entity.entity_id] = entity
        return True

    def get_entity(self, entity_id):
        return self._entities.get(entity_id)

    def find_entity_by_name(self, name, entity_type=None):
        for e in self._entities.values():
            if e.name == name:
                if entity_type is None or e.entity_type.value == entity_type:
                    return e
        return None

    def search(self, entity_type=None, limit=100):
        results = [
            e for e in self._entities.values()
            if entity_type is None or e.entity_type.value == entity_type
        ]
        return results[:limit], len(results)

    def upsert_entity(self, entity):
        self._entities[entity.entity_id] = entity
        return True


@pytest.fixture
def store():
    return MockStore()


@pytest.fixture
def bridge(store):
    return ParameterBridge(store)


@pytest.fixture
def builder(store):
    return ScenarioBuilder(store)


def _make_gene(store, symbol, gene_type="protein-coding"):
    g = Gene(name=symbol, symbol=symbol, gene_type=gene_type)
    store.add_entity(g)
    return g


def _make_drug(store, name, drug_class="", mechanism="", targets=None):
    d = Drug(
        name=name, drug_class=drug_class,
        mechanism=mechanism, targets=targets or [],
    )
    store.add_entity(d)
    return d


def _make_param_set(store, name, module, parameters, ranges=None):
    ps = ParameterSet(
        name=name, module=module, parameters=parameters,
        ranges=ranges or {},
    )
    store.add_entity(ps)
    return ps


# ── ParameterBridge Tests ────────────────────────────────────────────

class TestParameterBridge:

    def test_resolve_empty_scenario(self, bridge, store):
        """Empty scenario returns baseline defaults."""
        scenario = SimulationScenario(name="empty")
        store.add_entity(scenario)

        result = bridge.resolve_scenario(scenario)

        assert "cellular" in result
        assert result["cellular"]["n_normal_cells"] == 80
        assert result["immune"]["n_t_cells"] == 12

    def test_resolve_parameter_sets(self, bridge, store):
        """ParameterSets are loaded and merged by module."""
        ps1 = _make_param_set(store, "cell_params", "cellular", {
            "n_normal_cells": 50, "n_cancer_cells": 30,
        })
        ps2 = _make_param_set(store, "immune_params", "immune", {
            "n_t_cells": 25, "kill_probability": 0.9,
        })

        scenario = SimulationScenario(
            name="test",
            parameter_set_ids=[ps1.entity_id, ps2.entity_id],
        )
        store.add_entity(scenario)

        result = bridge.resolve_scenario(scenario)

        # ParameterSet values override baseline
        assert result["cellular"]["n_normal_cells"] == 50
        assert result["cellular"]["n_cancer_cells"] == 30
        assert result["immune"]["n_t_cells"] == 25
        assert result["immune"]["kill_probability"] == 0.9

    def test_drug_mapping_checkpoint_inhibitor(self, bridge, store):
        """Checkpoint inhibitor drug maps to immune module params."""
        drug = _make_drug(
            store, "Pembrolizumab",
            drug_class="checkpoint_inhibitor",
            targets=["PDCD1"],
        )

        scenario = SimulationScenario(
            name="test", drug_ids=[drug.entity_id],
        )
        store.add_entity(scenario)

        result = bridge.resolve_scenario(scenario)

        assert result["immune"]["n_t_cells"] == 30
        assert result["immune"]["kill_probability"] == 0.95

    def test_drug_mapping_anti_androgen(self, bridge, store):
        """Anti-androgen maps to ODE AR signaling params."""
        drug = _make_drug(
            store, "Enzalutamide",
            drug_class="anti-androgen",
            targets=["AR"],
        )

        scenario = SimulationScenario(
            name="test", drug_ids=[drug.entity_id],
        )
        store.add_entity(scenario)

        result = bridge.resolve_scenario(scenario)

        assert result["ode"]["system"] == "ar_signaling"
        assert result["ode"]["k_bind"] == 10.0

    def test_gene_mapping_specific(self, bridge, store):
        """Specific genes map to known parameter overrides."""
        tp53 = _make_gene(store, "TP53")
        myc = _make_gene(store, "MYC")

        scenario = SimulationScenario(
            name="test", gene_ids=[tp53.entity_id, myc.entity_id],
        )
        store.add_entity(scenario)

        result = bridge.resolve_scenario(scenario)

        # MYC should push transcription_rate high
        assert result.get("molecular", {}).get("transcription_rate", 0) >= 1.5
        # MYC division_time_cancer should be 6.0 (overrides TP53's 8.0)
        assert result.get("cellular", {}).get("division_time_cancer", 999) <= 8.0

    def test_gene_mapping_ar_selects_ode_system(self, bridge, store):
        """AR gene in scenario selects ar_signaling ODE system."""
        ar = _make_gene(store, "AR")

        scenario = SimulationScenario(
            name="test", gene_ids=[ar.entity_id],
        )
        store.add_entity(scenario)

        result = bridge.resolve_scenario(scenario)
        assert result.get("ode", {}).get("system") == "ar_signaling"

    def test_precedence_order(self, bridge, store):
        """Later layers override earlier ones."""
        # ParameterSet says n_t_cells = 5
        ps = _make_param_set(store, "low_immune", "immune", {"n_t_cells": 5})
        # Drug says n_t_cells = 30
        drug = _make_drug(
            store, "Pembrolizumab",
            drug_class="checkpoint_inhibitor",
        )

        scenario = SimulationScenario(
            name="test",
            parameter_set_ids=[ps.entity_id],
            drug_ids=[drug.entity_id],
        )
        store.add_entity(scenario)

        result = bridge.resolve_scenario(scenario)

        # Drug layer (4) overrides ParameterSet layer (2)
        assert result["immune"]["n_t_cells"] == 30

    def test_initial_conditions_override(self, bridge, store):
        """Initial cell counts override everything for matching params."""
        drug = _make_drug(store, "Pembro", drug_class="checkpoint_inhibitor")

        scenario = SimulationScenario(
            name="test",
            drug_ids=[drug.entity_id],
            initial_cell_counts={"t_cell": 50},
        )
        store.add_entity(scenario)

        result = bridge.resolve_scenario(scenario)

        # initial_cell_counts is highest priority
        assert result["immune"]["n_t_cells"] == 50

    def test_validation_warns_on_range_violation(self, bridge, store, caplog):
        """ParameterSet range violations are logged."""
        ps = _make_param_set(
            store, "bad_params", "cellular",
            {"n_normal_cells": 1000},
            ranges={"n_normal_cells": [1, 200]},
        )

        scenario = SimulationScenario(
            name="test", parameter_set_ids=[ps.entity_id],
        )
        store.add_entity(scenario)

        import logging
        with caplog.at_level(logging.WARNING):
            bridge.resolve_scenario(scenario)

        assert any("range violations" in msg for msg in caplog.messages)

    def test_validate_config_unknown_params(self, bridge):
        """validate_config flags unknown parameter names."""
        config = {
            "cellular": {"n_normal_cells": 80, "unknown_param": 42},
        }
        warnings = ParameterBridge.validate_config(config)
        assert any("unknown_param" in w for w in warnings)

    def test_merge_configs_multiple_layers(self):
        """_merge_configs correctly layers multiple dicts."""
        layer1 = {"cellular": {"a": 1, "b": 2}}
        layer2 = {"cellular": {"b": 3, "c": 4}}
        layer3 = {"immune": {"x": 10}}

        result = ParameterBridge._merge_configs(layer1, layer2, layer3)
        assert result["cellular"] == {"a": 1, "b": 3, "c": 4}
        assert result["immune"] == {"x": 10}

    def test_infer_enabled_modules(self, bridge, store):
        """infer_enabled_modules detects ODE/BNGL configs."""
        scenario = SimulationScenario(name="test")
        configs = {
            "cellular": {"n_normal_cells": 80},
            "ode": {"system": "ar_signaling"},
        }

        enabled = bridge.infer_enabled_modules(scenario, configs)
        assert enabled["cellular"] is True
        assert enabled.get("ode") is True


# ── ScenarioBuilder Tests ────────────────────────────────────────────

class TestScenarioBuilder:

    def test_build_scenario_basic(self, builder, store):
        """Build a basic scenario with genes and drugs."""
        _make_gene(store, "TP53")
        _make_gene(store, "AR")
        _make_drug(store, "Enzalutamide", drug_class="anti-androgen", targets=["AR"])

        scenario = builder.build_scenario(
            name="Test Scenario",
            scenario_type="drug_response",
            gene_names=["TP53", "AR"],
            drug_names=["Enzalutamide"],
        )

        assert scenario.name == "Test Scenario"
        assert scenario.scenario_type == "drug_response"
        assert len(scenario.gene_ids) == 2
        assert len(scenario.drug_ids) == 1

    def test_build_scenario_missing_entities_skipped(self, builder, store):
        """Missing entities are skipped without error."""
        _make_gene(store, "TP53")

        scenario = builder.build_scenario(
            name="Partial",
            gene_names=["TP53", "NONEXISTENT_GENE"],
        )

        assert len(scenario.gene_ids) == 1  # only TP53

    def test_build_prostate_cancer_scenario(self, builder, store):
        """Prostate cancer preset builds without error."""
        # Seed some genes
        for g in ["AR", "TP53", "PTEN", "MYC"]:
            _make_gene(store, g)

        scenario = builder.build_prostate_cancer_scenario()
        assert "Prostate Cancer" in scenario.name
        assert scenario.duration_hours == 72.0
        assert scenario.initial_cell_counts.get("normal") == 80

    def test_build_immunotherapy_scenario(self, builder, store):
        """Immunotherapy preset includes drug entity."""
        _make_gene(store, "PDCD1")
        _make_drug(store, "Pembrolizumab", drug_class="checkpoint_inhibitor")

        scenario = builder.build_immunotherapy_scenario("Pembrolizumab")
        assert "Immunotherapy" in scenario.name
        assert len(scenario.drug_ids) == 1

    def test_roundtrip_build_resolve(self, store):
        """ScenarioBuilder → ParameterBridge roundtrip works."""
        _make_gene(store, "AR")
        _make_drug(store, "Enzalutamide", drug_class="anti-androgen", targets=["AR"])

        builder = ScenarioBuilder(store)
        scenario = builder.build_scenario(
            name="Roundtrip",
            gene_names=["AR"],
            drug_names=["Enzalutamide"],
        )

        bridge = ParameterBridge(store)
        configs = bridge.resolve_scenario(scenario)

        assert "ode" in configs
        assert configs["ode"].get("system") == "ar_signaling"

    def test_list_available_genes(self, builder, store):
        """list_available_genes returns stored gene symbols."""
        _make_gene(store, "TP53")
        _make_gene(store, "BRCA1")

        genes = builder.list_available_genes()
        assert "TP53" in genes
        assert "BRCA1" in genes

    def test_auto_generate_params_creates_param_sets(self, builder, store):
        """auto_generate_params creates ParameterSet entities."""
        _make_gene(store, "AR")
        _make_drug(store, "Enzalutamide", drug_class="anti-androgen", targets=["AR"])

        scenario = builder.build_scenario(
            name="AutoParams",
            gene_names=["AR"],
            drug_names=["Enzalutamide"],
            auto_generate_params=True,
        )

        # Should have created ParameterSet entities
        assert len(scenario.parameter_set_ids) > 0

        # Verify they exist in store
        for ps_id in scenario.parameter_set_ids:
            ps = store.get_entity(ps_id)
            assert isinstance(ps, ParameterSet)
            assert ps.module in MODULE_PARAM_SCHEMA or ps.module == "ode"

    def test_build_immune_challenge(self, builder, store):
        """Immune challenge preset builds without error."""
        for g in ["TLR3", "IFNG", "CD3E", "HLA-A"]:
            _make_gene(store, g)

        scenario = builder.build_immune_challenge_scenario()
        assert "Immune Challenge" in scenario.name
        assert scenario.initial_cell_counts.get("t_cell") == 20

    def test_build_hypoxia_scenario(self, builder, store):
        """Hypoxia preset has low oxygen."""
        for g in ["HIF1A", "VEGFA"]:
            _make_gene(store, g)

        scenario = builder.build_hypoxia_scenario()
        assert scenario.initial_field_values.get("oxygen") == 0.05
