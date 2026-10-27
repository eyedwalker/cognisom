"""The cells a run draws are sized from the papers that measured them.

Until this pass the run exporter drew six of its seven lineages at a
``scale_um`` nobody measured, and ``lineage_appearance`` labelled each
one an estimate. ``seed_cell_size`` writes four measured radii (five
rows, since "NK cell" is two) from papers a verifier re-read, copies the
alveolar-macrophage measurement onto the M2 row with a rationale saying
that is what it is, and records why the luminal cell keeps its
placeholder.

The check is the same one the leukocyte pass set: each stored radius is
exactly half the number the paper prints, the rationale says what was
measured on what, and the renderer draws the measured diameter and says
where it came from.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library import seed_cell_size as seeding            # noqa: E402
from cognisom.library.parameters import Provenance, get_parameter  # noqa: E402


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


def _entity(store, name, entity_type=None):
    entity = store.find_entity_by_name(name, entity_type)
    assert entity is not None, (name, entity_type)
    return entity


def _pv(store, name, entity_type=None):
    value = get_parameter(_entity(store, name, entity_type), "radius_um")
    assert value is not None, f"{name} has no radius_um"
    return value


# ── every written value is half the paper's number ───────────────────

@pytest.mark.parametrize("name,entity_type,key", [
    (name, etype, key) for (name, etype), key in seeding.APPLIES.items()])
def test_each_radius_is_half_the_papers_diameter(store, name, entity_type, key):
    measured = seeding.BY_KEY[key]
    value = _pv(store, name, entity_type)
    assert value.value == pytest.approx(measured.diameter_um / 2.0, abs=0.001)
    assert value.unit == "um"
    assert value.provenance is Provenance.MEASURED
    assert value.citation == f"PMID:{measured.pmid}"
    assert measured.quote in value.rationale


@pytest.mark.parametrize("name,entity_type,key", [
    (name, etype, key) for (name, etype), key in seeding.APPLIES.items()])
def test_each_rationale_states_what_was_measured_on_what(store, name, entity_type, key):
    """Method, specimen, species, state, preparation, ruffles, quote:
    the six things a diameter cannot say for itself."""
    value = _pv(store, name, entity_type)
    for heading in ("METHOD:", "SPECIMEN:", "SPECIES:", "STATE (resting vs activated):",
                    "PREPARATION (suspension vs tissue section):",
                    "RUFFLE-INCLUSIVE:", "QUOTE:"):
        assert heading in value.rationale, (name, heading)
    assert "human" in value.rationale.lower()
    entity = _entity(store, name, entity_type)
    assert entity.sourced.get("diameter_convention") == seeding.BY_KEY[key].convention


# ── PMID 23626725: a circulating tumour cell, in suspension units ─────

def test_the_prostate_cancer_cell_is_the_coulter_calibrated_ctc_median(store):
    """10.7 um is Coumans' median prostate CTC diameter after the
    cartridge-flattened footprint is turned back into a suspension
    diameter (cylinder of thickness 5.2 um, fitted to a Coulter pipette).
    Not a tissue cell, and the record says so."""
    value = _pv(store, "Prostatic adenocarcinoma cell", "cell_type")
    assert value.value == pytest.approx(5.35)
    assert value.citation == "PMID:23626725"
    assert "cylinder" in value.rationale and "Coulter" in value.rationale
    assert "IMMC-38" in value.rationale
    assert "not a tumour cell in tissue" in value.conditions.lower() \
        or "not a tissue" in value.conditions.lower()
    # the two confirmed numbers it was preferred over, with their reasons
    assert "RUNNER-UP: PMID:24416373 7.97" in value.rationale
    assert "RUNNER-UP: PMID:32949099 10.3" in value.rationale
    assert "PROJECTED area" in value.rationale
    assert "underestimate" in value.rationale


# ── PMID 39273504: a 72 h blast measured in Fiji ─────────────────────

def test_the_cd8_effector_is_the_stimulated_row_of_table_a1(store):
    value = _pv(store, "CD8+ Effector T cell", "immune_cell")
    assert value.value == pytest.approx(5.65)
    assert value.citation == "PMID:39273504"
    assert "Fiji" in value.rationale and "15 cells" in value.rationale
    assert "CD3/CD28" in value.rationale
    assert "ACTIVATED" in value.rationale
    # the resting CD8 of the same table agrees with Downey's resting lymphocyte
    assert "6.8" in value.rationale and "PMID:2272970" in value.rationale
    assert "RUNNER-UP: PMID:32775494 11.04" in value.rationale
    assert "RUNNER-UP: PMID:31700012 9.22" in value.rationale


def test_the_effector_is_larger_than_the_resting_cell_it_came_from(store):
    """The internal check across two papers: a blast is bigger than the
    resting lymphocyte (Downey 1990, 6.05 um)."""
    effector = _pv(store, "CD8+ Effector T cell", "immune_cell")
    naive = _pv(store, "CD8+ Naive T cell", "immune_cell")
    assert naive.citation == "PMID:2272970"
    assert effector.value > naive.value * 1.5


# ── PMID 21103395: both NK rows, and the two numbers that were refuted ─

@pytest.mark.parametrize("entity_type", ["cell_type", "immune_cell"])
def test_both_nk_rows_take_the_primary_nk_diameter(store, entity_type):
    value = _pv(store, "NK cell", entity_type)
    assert value.value == pytest.approx(4.15)
    assert value.citation == "PMID:21103395"
    assert "IL-2" in value.rationale and "microwell" in value.rationale
    assert "data not shown" in value.rationale, "the weakness is on the record"
    assert "REFUTED: PMID:6166701" in value.rationale
    assert "REFUTED: PMID:7095831" in value.rationale
    assert "smear" in value.rationale


def test_the_two_nk_rows_agree(store):
    a = _pv(store, "NK cell", "cell_type")
    b = _pv(store, "NK cell", "immune_cell")
    assert a.to_dict() == b.to_dict()


# ── PMID 9400735: the M2 row copies M1 and says what the measurement is ─

def test_the_m2_macrophage_carries_the_m1_measurement_and_says_so(store):
    m1 = _pv(store, "Macrophage M1", "immune_cell")
    m2 = _pv(store, "Macrophage M2", "immune_cell")
    assert m1.citation == "PMID:9400735" and m1.provenance is Provenance.MEASURED
    assert m2.value == m1.value == pytest.approx(10.6)
    assert m2.citation == m1.citation
    assert m2.source == m1.source
    assert m2.provenance is Provenance.MEASURED
    assert "alveolar macrophages" in m2.rationale.lower()
    assert "polarization-specific size is unmeasured" in m2.rationale.lower()
    assert "Copied from the library's 'Macrophage M1' record" in m2.rationale
    assert "polarization-specific size is unmeasured" in m2.conditions.lower()


def test_the_copy_refuses_a_donor_that_is_not_a_measurement():
    """Copying launders nothing: a donor without a MEASURED radius against
    the expected PMID is refused, not copied."""
    import copy

    from cognisom.library.models import ImmuneCellEntity
    from cognisom.library.parameters import ParameterValue, set_parameter

    donor = ImmuneCellEntity(name="Macrophage M1")
    set_parameter(donor, "radius_um", ParameterValue(
        value=10.6, unit="um", provenance=Provenance.ESTIMATED,
        rationale="planted by a test"))
    target = ImmuneCellEntity(name="Macrophage M2")

    class _Store:
        def find_entity_by_name(self, name, entity_type=None):
            return {"Macrophage M1": donor, "Macrophage M2": target}.get(name)

        def update_entity(self, entity, changed_by=""):
            raise AssertionError("nothing should be written")

    before = copy.deepcopy(target.physics_params)
    run = seeding.seed(_Store())
    assert any("Macrophage M2" in line and "no MEASURED radius" in line
               for line in run.refused), run.refused
    assert target.physics_params == before


# ── what was deliberately not filled in ──────────────────────────────

def test_the_luminal_cell_keeps_its_placeholder_and_the_reason_is_recorded(store):
    """Three verified numbers, all heights of a columnar epithelium in a
    tissue section, no width, no suspension diameter: nothing to halve."""
    value = _pv(store, "Luminal epithelial cell", "cell_type")
    assert value.provenance is Provenance.PLACEHOLDER
    assert "Luminal epithelial cell" in seeding.DECLINED
    why = seeding.DECLINED["Luminal epithelial cell"]
    assert "HEIGHT" in why and "no confirmed DIAMETER" in why
    for pmid, um, _what in seeding.LUMINAL_HEIGHTS:
        assert f"PMID:{pmid}" in why and f"{um:g} um" in why
    assert "27.89" in why and "RWPE-1" in why


def test_nothing_declined_was_written(store):
    for name in seeding.DECLINED:
        entity = store.find_entity_by_name(name)
        assert entity is not None, name
        assert get_parameter(entity, "radius_um").provenance is Provenance.PLACEHOLDER


# ── the renderer draws the measured diameter and says so ─────────────

@pytest.mark.parametrize("name,entity_type,diameter", [
    ("Prostatic adenocarcinoma cell", "cell_type", 10.7),
    ("CD8+ Effector T cell", "immune_cell", 11.3),
    ("NK cell", "cell_type", 8.3),
    ("NK cell", "immune_cell", 8.3),
    ("Macrophage M2", "immune_cell", 21.2),
])
def test_resolved_scale_is_the_measured_diameter(store, name, entity_type, diameter):
    from cognisom.visualization.cell_scene import resolved_scale_um

    entity = _entity(store, name, entity_type)
    got, source = resolved_scale_um(entity)
    assert got == pytest.approx(diameter, abs=0.001)
    assert source.startswith("radius_um PMID:"), source
    # the stale second copy is still there and still disagrees; it must lose
    assert float(getattr(entity, "scale_um")) != pytest.approx(diameter)


def test_the_engine_lineages_are_now_cited_except_the_luminal_cell():
    """What this pass changes in the run exporter's record: four of the
    engine's lineages are cited, one is still openly an estimate."""
    from cognisom.visualization import lineage_appearance as la

    la.clear_cache()
    table = la.appearance_table()
    assert table["cancer"]["cited"] and table["cancer"]["diameter_um"] == pytest.approx(10.7)
    assert table["T_cell"]["cited"] and table["T_cell"]["diameter_um"] == pytest.approx(11.3)
    assert table["NK_cell"]["cited"] and table["NK_cell"]["diameter_um"] == pytest.approx(8.3)
    assert table["macrophage_M2"]["cited"] and table["macrophage_M2"]["diameter_um"] == pytest.approx(21.2)
    assert table["macrophage_M1"]["diameter_um"] == table["macrophage_M2"]["diameter_um"]
    assert table["macrophage_M1"]["mesh_type"] != table["macrophage_M2"]["mesh_type"], (
        "M1 and M2 are now the same size, so they must still differ by shape")
    assert not table["normal"]["cited"]
    assert table["normal"]["size_source"].startswith("estimate:")


def test_the_caption_does_not_call_a_measured_size_an_estimate(store):
    from cognisom.visualization.cell_scene import scene_for

    caption = scene_for(_entity(store, "NK cell", "immune_cell")).caption()
    assert "8.3 um" in caption and "(est)" not in caption


# ── the evidence travels with the repository ─────────────────────────

def test_the_curated_file_carries_every_value_this_pass_wrote():
    """A fresh clone, CI and production open a library seeded from
    curated_parameters.json; a value only in this machine's database is a
    value nobody else has."""
    from cognisom.library.curated import CURATED_PATH

    document = json.loads(CURATED_PATH.read_text())
    rows = {(e["name"], e["entity_type"]): e for e in document["entities"]}
    for (name, entity_type), key in seeding.APPLIES.items():
        record = rows[(name, entity_type)]["parameters"]["radius_um"]
        assert record["citation"] == f"PMID:{seeding.BY_KEY[key].pmid}", (name, entity_type)
        assert record["value"] == pytest.approx(seeding.BY_KEY[key].radius_um, abs=0.001)
        assert rows[(name, entity_type)]["sourced"]["diameter_convention"] == \
            seeding.BY_KEY[key].convention
    m2 = rows[("Macrophage M2", "immune_cell")]["parameters"]["radius_um"]
    assert m2["citation"] == "PMID:9400735" and m2["value"] == pytest.approx(10.6)
    # The luminal cell carries no evidenced value at all, so it has no
    # block in the file; if one ever appears it must not be a radius.
    luminal = rows.get(("Luminal epithelial cell", "cell_type"))
    assert luminal is None or "radius_um" not in luminal["parameters"]


def test_seeding_again_changes_nothing(store):
    """Idempotent: the pass can be re-run after a library rebuild."""
    before = {key: _pv(store, name, etype).to_dict()
              for (name, etype), key in seeding.APPLIES.items()}
    run = seeding.seed(store)
    assert not run.refused and not run.missing
    assert not run.corrected, run.corrected
    assert len(run.written) == len(seeding.APPLIES) + len(seeding.COPIED)
    assert len(run.declined) == len(seeding.DECLINED)
    after = {key: _pv(store, name, etype).to_dict()
             for (name, etype), key in seeding.APPLIES.items()}
    assert after == before
