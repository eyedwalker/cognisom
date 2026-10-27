"""The tissue engine sizes each cell by its archetype.

``TissueScaleConfig.cell_radius_um`` was serialised, documented and never
read: the distributor filled every cell with a literal 5.0 um, ghost cells
(a neighbour partition's boundary cells, as the force law sees them) got the
same literal, and no archetype could have its own size. Now:

- ``cell_radius_um`` is the radius of any archetype without its own;
- ``archetype_radius_um`` holds the rest, filled from the entity library by
  ``resolve_radii_from_library`` where the value is evidence -- first the
  macrophage's measured 10.6 um (PMID:9400735), since seed_cell_size also
  the cancer cell, the T cell and the NK cell -- and carried in the config
  JSON so a remote run sees the same sizes;
- a ghost carries its real radius;
- the mechanics bucket is sized to the largest cell. At the old fixed 15 um
  a cell above 6.5 um sent the neighbour search onto its fallback stencil,
  which misses contacts.

A default configuration is unchanged, cell for cell.
"""
from __future__ import annotations

import contextlib
import copy
import io
import logging

import numpy as np
import pytest

from cognisom.core.tissue_config import ARCHETYPE_ENTITY, TissueScaleConfig
from cognisom.core.tissue_engine import TissueSimulationEngine
from cognisom.genomics.parameter_resolver import (
    SimulationParameterResolver, reset_missing_log,
)
from cognisom.gpu.multi_gpu_mechanics import ADHESION_RANGE_UM, MechanicsConfig
from cognisom.library.parameters import (
    SIMULATION_GRADE, ParameterValue, Provenance, _sidecar, set_parameter,
)

MACROPHAGE_UM = 10.6


@pytest.fixture(autouse=True)
def _fresh_logs():
    reset_missing_log()
    yield
    reset_missing_log()


def _cfg(**kw):
    base = dict(n_cells=400, grid_shape=(20, 20, 20), tissue_size_um=(200.0, 200.0, 200.0),
                n_gpus=2, dt=0.05, duration=1.0, enable_ode=False,
                enable_diffusion=False, enable_cell_cycle=False)
    base.update(kw)
    return TissueScaleConfig(**base)


def _engine(cfg, **init):
    eng = TissueSimulationEngine(cfg)
    eng.initialize(**init)
    return eng


def _radii_by_type(eng):
    out = {}
    for ca in eng.cell_arrays:
        radii = eng._backend.to_numpy(ca.radii)[:ca.n_real]
        types = eng._backend.to_numpy(ca.cell_types)[:ca.n_real]
        for t, r in zip(types.tolist(), radii.tolist()):
            out.setdefault(t, set()).add(round(r, 4))
    return out


class _PlantedStore:
    def __init__(self, *entities):
        self._rows = {(e.name, e.entity_type.value): e for e in entities}

    def find_entity_by_name(self, name, entity_type=None):
        return self._rows.get((name, entity_type))


@pytest.fixture(scope="module")
def shipped_store():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore()


# ── the default is unchanged, and the config radius is live ─────────────

def _evidenced():
    """archetype -> measured radius, as the library answers today.

    The macrophage was the only one for a while; seed_cell_size added the
    cancer cell, the T cell and the NK cell. Read from the library rather
    than listed here, so the next curation pass lands without touching
    this file; what is asserted is that whatever is NOT evidenced falls
    back, and that the macrophage is still among the evidenced.
    """
    cfg = _cfg().resolve_radii_from_library()
    assert cfg.archetype_radius_um["macrophage"] == pytest.approx(MACROPHAGE_UM)
    return dict(cfg.archetype_radius_um)


def test_a_default_configuration_gives_five_microns_to_all_but_the_evidenced():
    """It used to give 5.0 to every archetype including the macrophage,
    because nothing resolved the library outside the dashboard. The
    engine resolves now, so every archetype with a measured radius gets
    it and everything else still falls back."""
    evidenced = _evidenced()
    assert set(evidenced) < set(_cfg().initial_composition), (
        "every archetype is evidenced; the fallback is no longer exercised")
    eng = _engine(_cfg())
    radii = {r for rs in _radii_by_type(eng).values() for r in rs}
    assert radii == {5.0} | {round(r, 4) for r in evidenced.values()}
    assert max(evidenced.values()) == pytest.approx(MACROPHAGE_UM), (
        "the bucket assertions below assume the macrophage is the largest cell")

    # And the neighbour bucket grew to match. This assertion used to read
    # `== MechanicsConfig().bucket_size_um`, i.e. the 15 um default, which
    # held only because nothing outside the dashboard resolved the
    # library. The engine's own comment says 15 um "covers 5 um cells and
    # nothing above 6.5 um" and that below the contact range the search
    # "falls back to the original 27-bucket stencil, misses included" --
    # so every non-dashboard run was missing macrophage contacts.
    assert eng._mechanics._config.bucket_size_um == pytest.approx(
        2.0 * MACROPHAGE_UM + ADHESION_RANGE_UM)
    assert eng._mechanics._config.bucket_size_um > \
        MechanicsConfig().bucket_size_um


def test_cell_radius_um_is_read_for_every_archetype_without_evidence():
    """cell_radius_um is the fallback, not a global override: a measured
    archetype keeps its measurement."""
    evidenced = _evidenced()
    cfg = _cfg(cell_radius_um=6.0)
    eng = _engine(cfg)
    by_type = _radii_by_type(eng)
    names = list(cfg.initial_composition)
    for index, radii in by_type.items():
        expected = round(evidenced.get(names[index], 6.0), 4)
        assert radii == {expected}, names[index]
    assert by_type[names.index("macrophage")] == {MACROPHAGE_UM}
    assert any(radii == {6.0} for radii in by_type.values()), "no archetype fell back"


# ── the library ─────────────────────────────────────────────────────────

def test_the_measured_macrophage_radius_reaches_the_cells():
    cfg = _cfg().resolve_radii_from_library()
    assert cfg.archetype_radius_um["macrophage"] == pytest.approx(MACROPHAGE_UM)
    assert cfg.archetype_radius_source["macrophage"] == "Macrophage: measured:PMID:9400735"
    by_type = _radii_by_type(_engine(cfg))
    names = list(cfg.initial_composition)
    for index, radii in by_type.items():
        assert radii == {round(cfg.radius_for(names[index]), 4)}, names[index]
    assert by_type[names.index("macrophage")] == {MACROPHAGE_UM}


def test_every_radius_the_library_supplied_is_evidence():
    """Holds whatever the library says: no archetype is sized by a stamp."""
    cfg = _cfg().resolve_radii_from_library()
    grades = {p.value for p in SIMULATION_GRADE}
    for archetype in cfg.archetype_radius_um:
        note = cfg.archetype_radius_source[archetype].split(": ", 1)[1]
        assert note.split(":")[0] in grades, (archetype, note)


def test_a_placeholder_radius_is_refused_and_a_missing_row_says_so(shipped_store):
    row = copy.deepcopy(shipped_store.find_entity_by_name("Luminal epithelial cell", "cell_type"))
    _sidecar(row).pop("radius_um", None)
    row.physics_params["radius_um"] = 7.5                     # a bare stamp
    cfg = _cfg().resolve_radii_from_library(SimulationParameterResolver(_PlantedStore(row)))
    assert "luminal_secretory" not in cfg.archetype_radius_um
    assert cfg.archetype_radius_source["luminal_secretory"] == (
        "Luminal epithelial cell: placeholder: not evidence")
    # smooth_muscle and cancer_epithelial used to report "no library
    # row" here, and that was the real state: ARCHETYPE_ENTITY mapped
    # eight of DEFAULT_COMPOSITION's ten, so a fifth of every simulated
    # tissue -- the malignant compartment among it -- had no entity at
    # all. Both are mapped now, so against this planted store they
    # report the row they looked for and did not find, which is the more
    # useful message of the two.
    for archetype, row in (("smooth_muscle", "Prostatic smooth muscle cell"),
                           ("cancer_epithelial", "Prostatic adenocarcinoma cell")):
        assert cfg.archetype_radius_source[archetype] == (
            f"{row}: no cell_type named {row!r}")


def test_an_archetype_with_no_mapping_still_says_so(shipped_store):
    """The other half of the message, which must survive every
    archetype in the shipped composition being mapped: a composition can
    name something ARCHETYPE_ENTITY has never heard of, and that has to
    read differently from a row that was looked up and missing."""
    composition = dict(_cfg().initial_composition)
    composition["an_archetype_nobody_mapped"] = 0.0
    cfg = _cfg(initial_composition=composition).resolve_radii_from_library(
        SimulationParameterResolver(shipped_store))
    assert cfg.archetype_radius_source["an_archetype_nobody_mapped"] == (
        "no library row")
    assert "an_archetype_nobody_mapped" not in cfg.archetype_radius_um


def test_a_curated_radius_on_any_mapped_archetype_is_used(shipped_store):
    """Prepared for the next curation pass."""
    name, etype = ARCHETYPE_ENTITY["fibroblast"]
    row = copy.deepcopy(shipped_store.find_entity_by_name(name, etype))
    _sidecar(row).pop("radius_um", None)
    assert set_parameter(row, "radius_um", ParameterValue(
        value=8.0, unit="um", provenance=Provenance.MEASURED,
        citation="PMID:12345678", source="planted by a test")) == []
    cfg = _cfg().resolve_radii_from_library(SimulationParameterResolver(_PlantedStore(row)))
    assert cfg.radius_for("fibroblast") == 8.0
    assert cfg.radius_for("basal") == cfg.cell_radius_um


def test_explicit_config_outranks_the_library():
    cfg = _cfg(archetype_radius_um={"macrophage": 8.0}).resolve_radii_from_library()
    assert cfg.radius_for("macrophage") == 8.0
    assert cfg.archetype_radius_source["macrophage"] == "config"


def test_the_radii_travel_with_the_config_json():
    cfg = _cfg(cell_radius_um=5.5).resolve_radii_from_library()
    back = TissueScaleConfig.from_json(cfg.to_json())
    assert back.cell_radius_um == 5.5
    assert back.archetype_radius_um == cfg.archetype_radius_um
    assert back.archetype_radius_source == cfg.archetype_radius_source


# ── the mechanics sees the large cells ──────────────────────────────────

def test_the_bucket_covers_the_largest_cell_and_no_contact_is_dropped(caplog):
    cfg = _cfg(enable_mechanics=True).resolve_radii_from_library()
    with caplog.at_level(logging.WARNING, logger="cognisom.gpu.multi_gpu_mechanics"):
        eng = _engine(cfg)
        eng.step()
    assert eng._mechanics._config.bucket_size_um >= 2 * MACROPHAGE_UM + ADHESION_RANGE_UM
    assert not eng._mechanics._warned_bucket_too_small
    assert not any("cannot see every contact" in r.getMessage() for r in caplog.records)


def _pair_force(bucket_um=None):
    """Contact force on two macrophages 19.9 um apart (contact at 21.2 um),
    placed so that a 15 um bucket puts them two buckets apart."""
    cfg = _cfg(n_gpus=1, initial_composition={"macrophage": 1.0}).resolve_radii_from_library()
    eng = TissueSimulationEngine(cfg)
    eng.initialize(cell_positions=np.array([[14.9, 50.0, 50.0], [34.8, 50.0, 50.0]]),
                   cell_types=np.zeros(2, dtype=np.int32))
    eng._mechanics._config.random_force = 0.0
    eng._mechanics._config.migration_force = 0.0
    if bucket_um is not None:
        eng._mechanics._config.bucket_size_um = bucket_um
    eng._mechanics.build_spatial_hash(eng.cell_arrays)
    return np.asarray(eng._mechanics.compute_forces(eng.cell_arrays, None)[0])[:2]


def test_two_overlapping_macrophages_feel_each_other():
    """With the bucket sized to them they repel: overlap 1.3 um x 2000 pN/um.

    Asserted on the force rather than on the positions after a step,
    because the integrator caps displacement at 1 um per substep: the pair
    is pushed out into the adhesion band, pulled back by the same 1 um, and
    after an even number of substeps sits exactly where it started -- a
    property of the capped integrator, not of the radius.
    """
    force = _pair_force()
    assert force[0, 0] == pytest.approx(-2600.0, rel=1e-3)
    assert force[1, 0] == pytest.approx(+2600.0, rel=1e-3)


def test_at_the_old_bucket_the_same_pair_was_invisible():
    """What the resized bucket prevents: at 15 um the fallback stencil
    never pairs two cells two buckets apart, however much they overlap."""
    assert np.allclose(_pair_force(bucket_um=15.0), 0.0)


def test_a_ghost_carries_its_real_radius():
    cfg = _cfg(initial_composition={"macrophage": 0.5, "basal": 0.5}).resolve_radii_from_library()
    eng = TissueSimulationEngine(cfg)
    # One macrophage just left of the split plane, one basal cell right of it.
    axis_mid = 100.0
    positions = np.array([[axis_mid - 4.0] * 3, [axis_mid + 4.0] * 3])
    eng.initialize(cell_positions=positions, cell_types=np.array([0, 1], dtype=np.int32))
    eng._distributor.update_ghosts(eng.cell_arrays)
    ghost_radii = sorted(
        r for ca in eng.cell_arrays
        for r in eng._backend.to_numpy(ca.radii)[ca.n_real:ca.n_real + ca.n_ghost].tolist())
    assert ghost_radii == pytest.approx(sorted([5.0, MACROPHAGE_UM]))


# ── every entry point, not just the dashboard ────────────────────────

def test_a_config_built_from_json_still_gets_the_measured_radius():
    """resolve_radii_from_library was called from the tissue-scale
    dashboard page and nowhere else in the package. The container
    entrypoint, tissue_runner.main, builds its config with from_json and
    never resolved -- so every non-dashboard run gave EVERY archetype the
    5.0 um default, macrophage included, whose measured radius is 10.6.

    The engine constructor is the one place every entry point passes
    through, so it resolves there.
    """
    import contextlib
    import io

    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    config = TissueScaleConfig.from_json(
        TissueScaleConfig.small_test().to_json())
    assert config.radius_for("macrophage") == pytest.approx(5.0), (
        "from_json already resolves; this test no longer proves anything")

    with contextlib.redirect_stdout(io.StringIO()):
        TissueSimulationEngine(config)

    assert config.radius_for("macrophage") == pytest.approx(10.6)
    assert "PMID:9400735" in config.archetype_radius_source["macrophage"]


def test_every_archetype_records_why_it_has_the_radius_it_has():
    """A run that silently used defaults for nine of ten archetypes is
    indistinguishable from one that resolved them. Each must say."""
    import contextlib
    import io

    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    config = TissueScaleConfig.from_json(
        TissueScaleConfig.small_test().to_json())
    with contextlib.redirect_stdout(io.StringIO()):
        TissueSimulationEngine(config)

    for archetype in config.initial_composition:
        assert config.archetype_radius_source.get(archetype), (
            f"{archetype} has a radius and no record of where it came from")


def test_resolving_twice_does_not_change_the_answer():
    """The dashboard still calls it explicitly; that must stay harmless."""
    import contextlib
    import io

    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    config = TissueScaleConfig.from_json(
        TissueScaleConfig.small_test().to_json())
    with contextlib.redirect_stdout(io.StringIO()):
        TissueSimulationEngine(config)
    first = dict(config.archetype_radius_um)

    config.resolve_radii_from_library()
    assert dict(config.archetype_radius_um) == first
