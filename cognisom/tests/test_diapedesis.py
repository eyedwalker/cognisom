"""
Tests for the Diapedesis Simulation Engine
==========================================

Covers:
- Entity database (adhesion molecules, gene sets, prototypes, schema)
- Simulation config and initialization
- State machine transitions
- Force models and boundary conditions
- Preset scenarios (LAD diseases)
- Full cascade verification
- Snapshot structure
- CPU fallback
"""

import math
import warnings

import numpy as np
import pytest

# Suppress numerical warnings during test
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Entity & Database Tests ────────────────────────────────────────────

class TestAdhesionMoleculeEntity:
    """Phase 1A: AdhesionMolecule entity type."""

    def test_entity_type_exists(self):
        from cognisom.library.models import EntityType
        assert hasattr(EntityType, "ADHESION_MOLECULE")
        assert EntityType.ADHESION_MOLECULE.value == "adhesion_molecule"

    def test_relationship_types_exist(self):
        from cognisom.library.models import RelationshipType
        assert hasattr(RelationshipType, "ADHERES_TO")
        assert hasattr(RelationshipType, "TRANSMIGRATES_THROUGH")
        assert hasattr(RelationshipType, "EXPRESSES_ON_SURFACE")

    def test_adhesion_molecule_roundtrip(self):
        from cognisom.library.models import AdhesionMolecule, EntityType
        am = AdhesionMolecule(
            name="E-selectin",
            entity_type=EntityType.ADHESION_MOLECULE,
            molecule_family="selectin",
            expressed_on=["endothelial"],
            ligands=["PSGL-1", "CD44"],
            binding_affinity_kd=100.0,
            on_rate=10.0,
            off_rate=5.0,
            regulation="cytokine_induced",
            diapedesis_step="rolling",
            gene_symbol="SELE",
            structure_type="type_I_transmembrane",
        )
        d = am.to_dict()
        am2 = AdhesionMolecule.from_dict(d)
        assert am2.molecule_family == "selectin"
        assert am2.binding_affinity_kd == 100.0
        assert am2.off_rate == 5.0
        assert am2.diapedesis_step == "rolling"
        assert am2.gene_symbol == "SELE"
        assert "PSGL-1" in am2.ligands

    def test_entity_class_map(self):
        from cognisom.library.models import ENTITY_CLASS_MAP, AdhesionMolecule
        assert "adhesion_molecule" in ENTITY_CLASS_MAP
        assert ENTITY_CLASS_MAP["adhesion_molecule"] is AdhesionMolecule


class TestDiapedesisGeneSet:
    """Phase 1B: DIAPEDESIS_ADHESION_GENES."""

    def test_gene_set_exists(self):
        from cognisom.library.gene_sets import DIAPEDESIS_ADHESION_GENES
        assert len(DIAPEDESIS_ADHESION_GENES) >= 35

    def test_key_genes_present(self):
        from cognisom.library.gene_sets import DIAPEDESIS_ADHESION_GENES
        key_genes = ["SELE", "SELP", "SELL", "ICAM1", "VCAM1", "PECAM1",
                     "ITGAL", "ITGB2", "CDH5", "SELPLG", "TNF", "IL1B"]
        for g in key_genes:
            assert g in DIAPEDESIS_ADHESION_GENES, f"{g} missing from gene set"

    def test_import_set_entry(self):
        from cognisom.library.gene_sets import IMPORT_SETS
        assert any("Diapedesis" in k for k in IMPORT_SETS.keys())


class TestSeedImmunologyDiapedesis:
    """Phase 1C: Seed adhesion molecules + endothelial cells."""

    def test_seed_counts(self):
        from cognisom.library.store import EntityStore
        from cognisom.library.seed_immunology import seed_immunology_catalog
        store = EntityStore()
        counts = seed_immunology_catalog(store)
        assert counts["adhesion_molecules"] == 15
        assert counts["endothelial_cells"] == 3

    def test_adhesion_molecules_valid(self):
        from cognisom.library.seed_immunology import _seed_adhesion_molecules
        from cognisom.library.models import AdhesionMolecule, EntityType

        class FakeStore:
            def __init__(self):
                self.entities = []
            def add_entity(self, e):
                self.entities.append(e)
                return e

        store = FakeStore()
        _seed_adhesion_molecules(store)

        for e in store.entities:
            assert e.entity_type == EntityType.ADHESION_MOLECULE
            assert e.molecule_family != ""
            assert e.diapedesis_step != ""
            assert e.gene_symbol != ""
            # Roundtrip
            d = e.to_dict()
            e2 = AdhesionMolecule.from_dict(d)
            assert e2.name == e.name


class TestDiapedesisPrototypes:
    """Phase 1D: Visualization prototypes."""

    def test_new_prototypes_registered(self):
        from cognisom.omniverse.prototype_library import ALL_PROTOTYPES
        expected = ["endothelial_cell", "selectin_molecule",
                    "integrin_low", "integrin_high", "ecm_fiber"]
        for name in expected:
            assert name in ALL_PROTOTYPES, f"{name} not in ALL_PROTOTYPES"

    def test_red_blood_cell_exists(self):
        from cognisom.omniverse.prototype_library import ALL_PROTOTYPES
        assert "red_blood_cell" in ALL_PROTOTYPES

    def test_total_prototype_count(self):
        from cognisom.omniverse.prototype_library import ALL_PROTOTYPES
        assert len(ALL_PROTOTYPES) >= 50


class TestBioEndothelialCell:
    """Phase 1E: Bio-USD schema."""

    def test_prim_registered(self):
        from cognisom.biousd.schema import prim_registry
        assert "bio_endothelial_cell" in prim_registry

    def test_create_prim(self):
        from cognisom.biousd.schema import create_prim, CellType
        ec = create_prim("bio_endothelial_cell",
                         cell_subtype="postcapillary_venule",
                         e_selectin_expression=0.8,
                         inflammation_state=0.6)
        assert ec.cell_subtype == "postcapillary_venule"
        assert ec.e_selectin_expression == 0.8
        assert ec.cell_type == CellType.ENDOTHELIAL

    def test_junction_integrity_default(self):
        from cognisom.biousd.schema import create_prim
        ec = create_prim("bio_endothelial_cell")
        assert ec.junction_integrity == 1.0


# ── Simulation Engine Tests ────────────────────────────────────────────

class TestDiapedesisConfig:
    """DiapedesisConfig defaults and validation."""

    def test_config_defaults(self):
        from cognisom.simulations.diapedesis import DiapedesisConfig
        cfg = DiapedesisConfig()
        assert cfg.vessel_length == 200.0
        assert cfg.vessel_radius == 25.0
        assert cfg.n_leukocytes == 20
        assert cfg.n_rbc == 200
        assert cfg.dt == 0.01
        assert 0 < cfg.inflammation_level <= 1.0

    def test_config_override(self):
        from cognisom.simulations.diapedesis import DiapedesisConfig
        cfg = DiapedesisConfig(vessel_radius=50.0, n_leukocytes=5)
        assert cfg.vessel_radius == 50.0
        assert cfg.n_leukocytes == 5


class TestDiapedesisSimInit:
    """Simulation initialization."""

    def test_initialization(self):
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        sim = DiapedesisSim(DiapedesisConfig(n_leukocytes=5, n_rbc=10))
        sim.initialize()
        assert sim._initialized
        assert sim.positions.shape == (15, 3)
        assert sim.leukocyte_states.shape == (5,)

    def test_vessel_geometry(self):
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        cfg = DiapedesisConfig(vessel_length=100.0, vessel_radius=20.0,
                               n_leukocytes=5, n_rbc=10)
        sim = DiapedesisSim(cfg)
        sim.initialize()

        # All particles should be inside vessel initially
        for i in range(15):
            r = math.sqrt(sim.positions[i, 1]**2 + sim.positions[i, 2]**2)
            assert r < cfg.vessel_radius, f"Particle {i} outside vessel: r={r}"

    def test_endothelial_cells_on_wall(self):
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        cfg = DiapedesisConfig(n_leukocytes=3, n_rbc=5)
        sim = DiapedesisSim(cfg)
        sim.initialize()

        for j in range(len(sim.endo_positions)):
            r = math.sqrt(sim.endo_positions[j, 1]**2 + sim.endo_positions[j, 2]**2)
            assert abs(r - cfg.vessel_radius) < 1.0, \
                f"Endothelial cell {j} not on wall: r={r}, R={cfg.vessel_radius}"


class TestFlowProfile:
    """Poiseuille flow velocity profile."""

    def test_parabolic_profile(self):
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        cfg = DiapedesisConfig(flow_velocity_max=500.0, vessel_radius=25.0,
                               n_leukocytes=1, n_rbc=1)
        sim = DiapedesisSim(cfg)
        sim.initialize()

        # Center: maximum velocity
        v_center = sim._flow_velocity_at(np.array([50.0, 0.0, 0.0]))
        assert abs(v_center - 500.0) < 1.0

        # Half radius: 75% of max
        v_half = sim._flow_velocity_at(np.array([50.0, 12.5, 0.0]))
        assert abs(v_half - 375.0) < 1.0

        # Wall: zero
        v_wall = sim._flow_velocity_at(np.array([50.0, 25.0, 0.0]))
        assert abs(v_wall) < 1.0


class TestStateTransitions:
    """Leukocyte state machine transitions."""

    def test_initial_state_flowing(self):
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig, LeukocyteState
        sim = DiapedesisSim(DiapedesisConfig(n_leukocytes=10, n_rbc=5))
        sim.initialize()
        assert all(s == LeukocyteState.FLOWING for s in sim.leukocyte_states)

    def test_full_cascade_severe(self):
        """In severe inflammation, at least 1 leukocyte should migrate."""
        from cognisom.simulations.diapedesis import DiapedesisSim, LeukocyteState
        sim = DiapedesisSim.severe_inflammation()
        sim.initialize()
        frames = sim.run(duration=120.0, fps=5)
        final_states = frames[-1]["leukocyte_states"]
        n_migrated = sum(1 for s in final_states if s == LeukocyteState.MIGRATED)
        assert n_migrated >= 1, f"Expected ≥1 migrated, got {n_migrated}"

    def test_metrics_tracking(self):
        from cognisom.simulations.diapedesis import DiapedesisSim
        sim = DiapedesisSim.severe_inflammation()
        sim.initialize()
        frames = sim.run(duration=30.0, fps=5)
        m = frames[-1]["metrics"]
        assert "state_counts" in m
        assert "avg_rolling_velocity" in m
        assert "n_migrated" in m
        assert "time" in m
        assert m["time"] > 0


class TestPresetScenarios:
    """LAD disease scenarios and healthy vessel."""

    def test_lad1_no_arrest(self):
        """LAD-1: leukocytes roll but never firmly arrest."""
        from cognisom.simulations.diapedesis import DiapedesisSim
        sim = DiapedesisSim.lad1_no_lfa1()
        sim.initialize()
        frames = sim.run(duration=60.0, fps=5)
        m = frames[-1]["metrics"]["state_counts"]
        assert m["arrested"] == 0, f"LAD-1 should have 0 arrested, got {m['arrested']}"
        assert m["migrated"] == 0, f"LAD-1 should have 0 migrated, got {m['migrated']}"

    def test_lad2_no_rolling(self):
        """LAD-2: no selectin binding → no rolling at all."""
        from cognisom.simulations.diapedesis import DiapedesisSim
        sim = DiapedesisSim.lad2_no_selectin_ligand()
        sim.initialize()
        frames = sim.run(duration=60.0, fps=5)
        m = frames[-1]["metrics"]["state_counts"]
        assert m["rolling"] == 0, f"LAD-2 should have 0 rolling, got {m['rolling']}"
        assert m["arrested"] == 0

    def test_lad3_no_integrin_activation(self):
        """LAD-3: rolling occurs but integrin never activates."""
        from cognisom.simulations.diapedesis import DiapedesisSim
        sim = DiapedesisSim.lad3_no_kindlin3()
        sim.initialize()
        frames = sim.run(duration=60.0, fps=5)
        m = frames[-1]["metrics"]
        assert m["state_counts"]["migrated"] == 0

    def test_healthy_vessel(self):
        """Healthy vessel: minimal recruitment."""
        from cognisom.simulations.diapedesis import DiapedesisSim
        sim = DiapedesisSim.healthy_vessel()
        sim.initialize()
        frames = sim.run(duration=60.0, fps=5)
        m = frames[-1]["metrics"]["state_counts"]
        n_migrated = m["migrated"]
        # In healthy vessel (inflammation=0.1), very few should migrate
        assert n_migrated <= 5, f"Healthy vessel: expected ≤5 migrated, got {n_migrated}"


class TestSnapshotStructure:
    """Verify snapshot dict has all required fields."""

    def test_snapshot_keys(self):
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        sim = DiapedesisSim(DiapedesisConfig(n_leukocytes=3, n_rbc=5))
        sim.initialize()
        sim.step()
        snap = sim.get_snapshot()

        required = [
            "time", "step",
            "leukocyte_positions", "leukocyte_colors", "leukocyte_radii",
            "leukocyte_states", "integrin_activation", "transmigration_progress",
            "rbc_positions", "rbc_colors",
            "endo_positions", "endo_colors",
            "endo_selectin_expr", "endo_junction_integrity",
            "vessel_length", "vessel_radius", "metrics",
        ]
        for key in required:
            assert key in snap, f"Missing snapshot key: {key}"

    def test_snapshot_array_lengths(self):
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        cfg = DiapedesisConfig(n_leukocytes=5, n_rbc=10)
        sim = DiapedesisSim(cfg)
        sim.initialize()
        sim.step()
        snap = sim.get_snapshot()

        assert len(snap["leukocyte_positions"]) == 5
        assert len(snap["leukocyte_colors"]) == 5
        assert len(snap["rbc_positions"]) == 10


class TestBoundaryEnforcement:
    """Vessel boundary conditions."""

    def test_rbc_stays_inside(self):
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        cfg = DiapedesisConfig(n_leukocytes=2, n_rbc=20)
        sim = DiapedesisSim(cfg)
        sim.initialize()

        for _ in range(100):
            sim.step()

        # All RBCs inside vessel
        for ri in sim.rbc_indices:
            r = math.sqrt(sim.positions[ri, 1]**2 + sim.positions[ri, 2]**2)
            assert r < cfg.vessel_radius + 1.0, \
                f"RBC {ri} outside vessel: r={r}"

    def test_x_periodic(self):
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        cfg = DiapedesisConfig(n_leukocytes=2, n_rbc=5)
        sim = DiapedesisSim(cfg)
        sim.initialize()

        for _ in range(500):
            sim.step()

        for i in range(len(sim.positions)):
            x = sim.positions[i, 0]
            assert 0 <= x <= cfg.vessel_length + 1.0, \
                f"Particle {i} x={x} outside [0, {cfg.vessel_length}]"


class TestCPUFallback:
    """Verify simulation works without Warp."""

    def test_gpu_module_loads(self):
        from cognisom.physics.diapedesis_kernels import DiapedesisGPU
        gpu = DiapedesisGPU()
        # Should not crash even without GPU
        assert isinstance(gpu.available, bool)

    def test_simulation_runs_cpu_only(self):
        """Full simulation should work on CPU (NumPy) path."""
        from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        sim = DiapedesisSim(DiapedesisConfig(n_leukocytes=3, n_rbc=5))
        sim.initialize()
        frames = sim.run(duration=5.0, fps=5)
        assert len(frames) > 0
        assert not np.any(np.isnan(np.array(frames[-1]["leukocyte_positions"])))


class TestChemokineField:
    """Chemokine diffusion field."""

    def test_field_initialization(self):
        from cognisom.simulations.diapedesis import ChemokineField
        field = ChemokineField(200.0, 25.0)
        assert field.concentration.shape == (40, 20)

    def test_source_and_diffusion(self):
        from cognisom.simulations.diapedesis import ChemokineField
        field = ChemokineField(200.0, 25.0)
        field.add_source(100.0, 1.0)

        # Run diffusion
        for _ in range(200):
            field.update(0.1)

        # Concentration should be non-zero somewhere
        assert np.max(field.concentration) > 0.0

    def test_gradient(self):
        from cognisom.simulations.diapedesis import ChemokineField
        field = ChemokineField(200.0, 25.0)
        field.add_source(100.0, 1.0)
        for _ in range(100):
            field.update(0.1)

        # Gradient should exist near source
        g = field.get_gradient_x_at(100.0, 25.0)
        # Just check it doesn't crash and returns finite value
        assert np.isfinite(g)
