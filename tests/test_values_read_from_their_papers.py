"""Values a person read out of the paper, and what the reading changed.

Every value in this library graded MEASURED carries a PMID, and the
grade asserts somebody read the measurement. For most of them nobody
had. These are the ones where somebody now has, and the record is
worth stating plainly: of five papers obtained so far, three changed a
stored number and two confirmed one.

The failure mode is never a missing citation. It is a citation that
points at a real paper containing a real number that is not the number
you want -- a median across species, a different enzyme, a
non-physiological buffer.
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


def _pv(store, name, parameter):
    from cognisom.library.parameters import get_parameter

    entity = store.find_entity_by_name(name)
    assert entity is not None, name
    value = get_parameter(entity, parameter)
    assert value is not None, f"{name} has no {parameter}"
    return value


# ── PMID 15209560: a median across three different enzymes ───────────

def test_the_caspase8_kcat_is_the_homodimer_in_physiological_salt(store):
    """It was 0.88, DERIVED, "the median of 4 BRENDA measurements...
    spanning 0.65 to 1.25". Table 1 shows those four are not four
    measurements of one thing:

        0.65  cleaved caspase-8 homodimer, low salt
        1.25  the SAME enzyme in 0.7 M sodium citrate
        0.98  caspase-8/FLIP-L HETERODIMER, 0.7 M citrate
        0.78  the same heterodimer with FLIP-L D297A

    Three molecular species, three of them in a kosmotrope that the
    paper uses as a tool to force dimerization. A median across that is
    not a property of caspase-8.
    """
    from cognisom.library.parameters import Provenance

    value = _pv(store, "Caspase-8", "kcat_per_s")
    assert value.value == pytest.approx(0.65)
    assert value.provenance is Provenance.MEASURED
    assert value.citation == "PMID:15209560"
    assert "0.7 M sodium citrate" in value.rationale
    assert "heterodimer" in value.rationale.lower()
    # the buffer it was measured in is named, because that is the whole
    # distinction between 0.65 and 1.25
    assert "20 mM Tris" in value.conditions


def test_the_km_names_the_peptide_it_was_measured_against(store):
    """55.2 uM is the Km for Ac-IETD-AFC, a fluorogenic TETRAPEPTIDE. It
    is not the Km for BID or procaspase-3, and the stored name has to
    say so, or the next reader takes a peptide constant for a protein
    one."""
    from cognisom.library.parameters import km_substrate

    entity = store.find_entity_by_name("Caspase-8")
    assert km_substrate(entity) == "Ac-IETD-AFC peptide"


def test_the_km_comes_from_the_same_row_as_the_kcat(store):
    """A kcat and a Km only belong together if one enzyme produced both
    under one condition. These are the same line of Table 1, and the
    check is that they reproduce the kcat/Km the paper reports."""
    kcat = _pv(store, "Caspase-8", "kcat_per_s")
    km = _pv(store, "Caspase-8", "km_um")
    assert km.value == pytest.approx(55.2)
    assert km.citation == kcat.citation

    efficiency = kcat.value / (km.value * 1e-6)
    assert efficiency == pytest.approx(1.2e4, rel=0.05), (
        "the stored pair does not reproduce the paper's reported "
        f"kcat/Km of 1.2e4 M^-1 s^-1; got {efficiency:.2e}")


# ── PMID 16840338: confirmed, and the neighbours named ───────────────

def test_the_influenza_eclipse_is_confirmed_and_its_neighbours_named(store):
    """6.0 h survives contact with its paper -- 'it takes about 6 h
    between infection of target cells and virion release'. What the
    record gained is the two quantities sitting beside it in the same
    table that a reader could take by mistake."""
    value = _pv(store, "Influenza A", "replication_cycle_hours")
    assert value.value == pytest.approx(6.0)
    assert value.citation == "PMID:16840338"
    assert "4.6-7.9" in value.conditions, "the 95% CI is not recorded"
    assert "upper" in value.conditions.lower()
    # the two it is not
    assert "4.6 h" in value.rationale and "11.4 h" in value.rationale


# ── PMID 2272970: the sizes, and a citation written before the read ──

RESTING = ("Naive B cell", "Memory B cell",
           "CD8+ Naive T cell", "CD8+ Memory T cell")


def test_the_neutrophil_radius_came_from_the_column_the_authors_trust(store):
    """It read 4.2, graded MEASURED against this PMID, before anyone had
    the paper. 4.2 is the Coulter mean; the authors lead with flow
    cytometry and name a bias in the Coulter figure. Nothing recorded
    which of the two columns it was."""
    from cognisom.library.parameters import Provenance

    value = _pv(store, "Neutrophil", "radius_um")
    assert value.value == pytest.approx(4.167, abs=0.001)
    assert value.provenance is Provenance.MEASURED
    assert value.citation == "PMID:2272970"
    # both methods on the record, so the choice is visible
    assert "8.33" in value.conditions and "8.40" in value.conditions
    assert "COULTER" in value.rationale


def test_the_monocyte_was_nearly_twice_its_real_size(store):
    """7.5 um radius is a 15 um cell. The measurement is 8.13 um across
    -- a factor of 1.85 in radius, 6.3 in volume, under every surface
    area and every concentration derived from it."""
    value = _pv(store, "Monocyte", "radius_um")
    assert value.value == pytest.approx(4.063, abs=0.001)
    assert value.citation == "PMID:2272970"


@pytest.mark.parametrize("name", RESTING)
def test_only_the_resting_lymphocytes_take_the_bulk_measurement(store, name):
    """Downey measured a 98-99% pure elutriated lymphocyte fraction: a
    small resting cell. A naive or resting memory cell is one."""
    value = _pv(store, name, "radius_um")
    assert value.value == pytest.approx(3.025, abs=0.001)
    assert "BULK" in value.rationale


@pytest.mark.parametrize("name", [
    "CD4+ Th1 cell", "CD8+ Effector T cell", "NK cell", "Plasma cell"])
def test_the_effector_subsets_were_left_alone(store, name):
    """A blasting effector is not a small resting lymphocyte, so these
    keep their placeholders. Filling them from this paper would have
    been the easy sixteen extra values and all of them wrong."""
    from cognisom.library.parameters import Provenance
    from cognisom.library.seed_leukocyte_size import DECLINED

    value = _pv(store, name, "radius_um")
    assert value.provenance is Provenance.PLACEHOLDER
    assert name in DECLINED, "left unfilled without recording why"


def test_the_transcribed_table_reproduces_the_papers_own_text(store):
    """The paper rounds its flow means to 8.3 / 8.13 / 6.05 in prose.
    Recomputing them from the transcribed donor rows is the check that
    the table was read correctly."""
    from cognisom.library.seed_leukocyte_size import BY_LINEAGE

    assert BY_LINEAGE["neutrophil"].flow_diameter == pytest.approx(8.33, 1e-3)
    assert BY_LINEAGE["monocyte"].flow_diameter == pytest.approx(8.13, 1e-3)
    assert BY_LINEAGE["lymphocyte"].flow_diameter == pytest.approx(6.05, 1e-9)


# ── the two sizes, and which one the renderer draws ──────────────────

def test_a_sourced_radius_beats_an_unprovenanced_scale(store):
    """An entity can carry two sizes. ``scale_um`` has no provenance and
    disagreed with every measured diameter by 1.16x to 1.85x, because it
    was 2 x radius back when the radius was a guess and nobody updated
    it when a real one arrived. The one with a citation wins."""
    from cognisom.visualization.cell_scene import resolved_scale_um

    entity = store.find_entity_by_name("Monocyte")
    assert float(getattr(entity, "scale_um")) == pytest.approx(15.0), (
        "the stale second copy is gone; this test no longer proves "
        "anything about preferring the sourced one")
    diameter, source = resolved_scale_um(entity)
    assert diameter == pytest.approx(8.126, abs=0.001)
    assert source.startswith("radius_um")


def test_an_unsourced_size_is_still_used_and_labelled(store):
    """Nothing is invented for a cell this paper did not measure: it
    keeps drawing at scale_um, and the caption marks it an estimate."""
    from cognisom.visualization.cell_scene import resolved_scale_um, scene_for

    entity = store.find_entity_by_name("NK cell")
    _diameter, source = resolved_scale_um(entity)
    assert source == "scale_um (no provenance)"
    assert "(est)" in scene_for(entity).caption()


def test_the_caption_does_not_call_a_sourced_size_an_estimate(store):
    """``scene.sources`` was assigned wholesale further down, which
    dropped the scale provenance set at the top and captioned a measured
    diameter as a guess."""
    from cognisom.visualization.cell_scene import scene_for

    caption = scene_for(store.find_entity_by_name("Monocyte")).caption()
    assert "8.126 um" in caption and "(est)" not in caption


# ── the papers are on disk and readable ──────────────────────────────

@pytest.mark.parametrize("pmid,probe", [
    ("15209560", "0.65"),
    ("16840338", "6.0"),
    ("15130947", "4.1"),
    ("27647916", "70"),
    ("6775712", "44.4"),
    ("2272970", "8.34"),
])
def test_each_paper_is_readable_and_contains_its_number(pmid, probe):
    """A PDF that cannot be re-read is a citation again. These parse,
    and the number each one holds up is findable in the text."""
    from cognisom.library import literature as lit

    if not (REPO_ROOT / "docs" / "papers" / f"{pmid}.pdf").exists():
        pytest.skip(f"{pmid}.pdf is not on this machine")
    article = lit.fetch(pmid)
    assert article.full_text, f"{pmid} did not parse"
    assert article.route == "local:pdf"
    assert probe in article.text, f"{probe} not found in {pmid}"


def test_the_papers_directory_is_still_not_committed():
    """Licensed copies. Not ours to redistribute."""
    assert "docs/papers" in (REPO_ROOT / ".gitignore").read_text()
