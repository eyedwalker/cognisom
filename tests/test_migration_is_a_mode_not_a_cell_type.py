"""0.234 um/min was right, and was the wrong regime for a tumour.

PMID 25679760 measures the same cells at three speeds:

    mesenchymal (2D, adherent, unconfined)   0.234 +/- 0.09 um/min
    A1 amoeboid (confined, low adhesion)     1.7   +/- 0.4
    A2 amoeboid ("stable bleb")              5.3   +/- 1.5

Twenty-three fold, no genetic change. The trigger is 3 um of
confinement plus the absence of focal adhesions -- and the authors'
closing argument is that the confinement a growing tumour imposes on
itself may be enough to cause it.

The library stored the unconfined number with conditions taken from a
BioNumbers record and a rationale quoting a paragraph about epithelial
cells. It also had no way to turn a speed into the cell diffusion
coefficient the mechanics needs. The same paper supplies that too.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


def _speed(store):
    from cognisom.library.parameters import get_parameter

    entity = store.find_entity_by_name("Stromal fibroblast")
    assert entity is not None
    return entity, get_parameter(entity, "migration_speed_um_per_min")


def test_the_stored_speed_is_unchanged_and_now_says_which_regime(store):
    from cognisom.library.parameters import Provenance

    _entity, pv = _speed(store)
    assert pv.value == pytest.approx(0.234)
    assert pv.provenance is Provenance.MEASURED
    assert pv.citation == "PMID:25679760"
    assert "THE SPEED IS THE MODE, NOT THE CELL" in pv.conditions
    assert "UNCONFINED" in pv.conditions


def test_the_other_two_regimes_travel_with_it(store):
    """A reader who sees only 0.234 cannot know the same cell reaches
    5.3 once confined."""
    entity, _pv = _speed(store)
    modes = (getattr(entity, "sourced", None) or {}).get("migration_modes")
    assert modes and set(modes) == {"mesenchymal", "amoeboid_a1",
                                    "amoeboid_a2"}
    assert modes["amoeboid_a2"]["speed_um_per_min"] == pytest.approx(5.3)
    assert modes["amoeboid_a1"]["speed_um_per_min"] == pytest.approx(1.7)


def test_the_confinement_trigger_is_recorded(store):
    """The switch is not spontaneous: it needs confinement AND the
    absence of focal adhesions. Without that, the three numbers look
    like three cell types."""
    _entity, pv = _speed(store)
    assert "confinement" in pv.conditions.lower()
    assert "focal adhesion" in pv.conditions.lower()
    assert "tumour" in pv.conditions.lower() or \
        "tumor" in pv.conditions.lower()


def test_the_bionumbers_rationale_about_epithelial_cells_is_gone(store):
    _entity, pv = _speed(store)
    assert "BioNumbers quantity" not in pv.conditions
    assert "EPITHELIAL" in pv.rationale, (
        "the record should say what the old rationale actually quoted")


# ── the conversion the mechanics needed ──────────────────────────────

def test_diffusion_needs_persistence_as_well_as_speed():
    """D = S^2 P / 2. A speed alone does not give a diffusion
    coefficient: a fast cell that turns constantly goes nowhere."""
    from cognisom.library.seed_migration_modes import \
        diffusion_coefficient_um2_per_min as D

    assert D(0.234, 60.0) == pytest.approx(0.234 ** 2 * 60.0 / 2.0)
    # same speed, a tenth of the persistence -> a tenth of the diffusion
    assert D(0.234, 6.0) == pytest.approx(D(0.234, 60.0) / 10.0)
    # and it is quadratic in speed, which is why the mode matters so much
    assert D(5.3, 10.0) / D(0.234, 10.0) == pytest.approx((5.3 / 0.234) ** 2)


def test_the_walk_is_ballistic_below_the_persistence_time():
    """Below P the Fuerth model is ballistic, so a tissue step shorter
    than P is in a regime where a plain random walk is wrong."""
    from cognisom.library.seed_migration_modes import \
        mean_square_displacement_um2 as msd

    speed, persistence = 5.3, 10.0
    short = msd(speed, persistence, 0.5)
    ballistic = (speed * 0.5) ** 2
    assert short == pytest.approx(ballistic, rel=0.1)

    # far above P it is diffusive. In TWO dimensions MSD -> 4 D t, and
    # that factor is what pins D = S^2 P / 2 rather than S^2 P.
    from cognisom.library.seed_migration_modes import \
        diffusion_coefficient_um2_per_min as D
    long_t = 10_000.0
    assert msd(speed, persistence, long_t) == pytest.approx(
        4.0 * D(speed, persistence) * long_t, rel=0.01)


def test_the_model_has_the_units_of_an_area():
    """A single S instead of S^2 gives um.min, which is how the slip
    would have gone unnoticed -- the number still looks plausible."""
    from cognisom.library.seed_migration_modes import \
        mean_square_displacement_um2 as msd

    # doubling the speed must quadruple the displacement squared
    assert msd(2.0, 10.0, 5.0) / msd(1.0, 10.0, 5.0) == pytest.approx(4.0)


def test_a_negative_input_is_refused():
    from cognisom.library.seed_migration_modes import \
        diffusion_coefficient_um2_per_min as D

    with pytest.raises(ValueError):
        D(-1.0, 10.0)


def test_the_paper_is_on_disk_and_holds_all_three_speeds():
    from cognisom.library import literature as lit

    if not (REPO_ROOT / "docs" / "papers" / "25679760.pdf").exists():
        pytest.skip("25679760.pdf is not on this machine")
    text = lit.fetch("25679760").text
    for probe in ("0.234", "5.3", "1.7 "):
        assert probe in text
    assert "Furth Model" in text
