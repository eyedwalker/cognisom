"""The obvious data source for MHC-I cannot carry it, and still cannot.

WRITTEN 2026-10-22: the column is now filled by
``core.antigen_presentation`` from labelled estimates. Everything below is
still true and is why those numbers are estimates rather than measurements.

``mhc1`` is column 6 of the tissue cell-state array and nothing writes it. On a
cancer immunology platform that is the variable the whole
antigen-presentation story runs through: whether a CD8 T cell can see a tumour
cell at all.

The consumers already exist, which is why this is worth stating rather than
just leaving on a list. ``cell_renderer`` accepts ``color_by="mhc1"`` and
prints "MHC-I: {value}" in its hover text, ``exporters`` writes an
``mhc1_expression`` column, and the renderer's own demo data encodes the
biology it expects -- 0.9 for normal cells, 0.3 for cancer, 1.0 for immune --
with synthetic noise. So the display path is built and fed by invented
numbers, and the gap is entirely producer-side.

**The library's B2M data looks like the answer and is not.** B2M is the
invariant light chain of every MHC class I molecule, so its abundance is the
natural proxy, and the library carries ``abundance_ppm`` for it across 20
immune cell types, sourced to PaxDb v6.1 / Kim et al., Nature 2014. But those
20 values are 5 measurements, each from a different bulk sample, broadcast
across the subsets of each lineage -- which ``abundance_measured_in`` records,
and which is the only reason it is visible. Worse, the implied ordering is not
credible: CD4+ T cells come out at 2.79x CD8+ T cells, from two different bulk
samples, for a chain that is present on every nucleated cell. That is within
the between-sample variation of spectral counting on a draft proteome. Wiring
it in would encode a measurement artefact as a 2.8x difference in how visible
a helper T cell is to a killer.

**What would actually close it**, neither of which this repo has yet:

* A cancer-versus-normal MHC-I level. That is the number immune evasion turns
  on, and the library's B2M record covers immune cells only -- there is no
  tumour value in it at all.
* A quantitative hypoxia coefficient. The direction is solid and
  mechanistically over-determined: hypoxia suppresses MHC-I HIF-dependently
  and also downregulates TAP1/2 and LMP7, which dominate surface display of
  peptide-MHC complexes (Sethumadhavan et al., PLoS One 2017, PMID 29155844).
  Neither that paper's abstract nor the later autophagy/PERK and HIF1A-AS2
  accounts give a fold change, so any coefficient here would be invented --
  and this is not a variable to put a made-up number into, because it sets the
  simulated rate of immune escape directly.

One detail from PMID 29155844 is worth keeping for when that coefficient
arrives: the effect shows up in 3-dimensional cultures and NOT in 2D. It is
only observable where real oxygen gradients exist, so it is a coupling this
model could not have expressed before the oxygen field became trustworthy,
and a monolayer model cannot express at all.
"""

import numpy as np
import pytest

COLUMNS = ["oxygen", "glucose", "atp", "lactate", "age", "phase", "mhc1",
           "division_timer", "damage", "apoptotic_signal",
           "migration_speed", "reserved"]
MHC1 = COLUMNS.index("mhc1")


@pytest.fixture(scope="module")
def b2m():
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore

        store = EntityStore()
        entities, _ = store.search(limit=10000)
    matches = [e for e in entities if (e.name or "").startswith("B2M")]
    assert matches, "the library no longer has a B2M record"
    return matches[0].to_dict()["properties"]


def test_the_library_does_carry_sourced_b2m_abundance(b2m):
    """So the reason not to use it has to be about the data, not its absence."""
    assert b2m["gene_symbol"] == "B2M"
    assert b2m["mhc_class"] == "I"
    assert "all_nucleated" in b2m["tissue_distribution"]
    assert b2m["abundance_ppm"], "no abundance to argue about"
    assert "PaxDb" in b2m["abundance_source"]
    assert "nature13302" in b2m["abundance_source"], (
        "the measurement should still cite Kim et al. 2014")


def test_twenty_cell_types_are_five_measurements(b2m):
    """Each lineage's subsets share one bulk sample's value.

    This is the fact that makes the data unusable for per-cell-type levels,
    and it is only visible because the library kept
    ``abundance_measured_in`` alongside the numbers.
    """
    ppm = b2m["abundance_ppm"]
    measured_in = b2m["abundance_measured_in"]
    assert set(ppm) == set(measured_in), (
        "every abundance must say which sample it came from")

    assert len(ppm) >= 20
    assert len(set(measured_in.values())) == 5, sorted(
        set(measured_in.values()))
    # One value per sample: the broadcast is exact, not approximate.
    for sample in set(measured_in.values()):
        values = {ppm[cell] for cell, src in measured_in.items()
                  if src == sample}
        assert len(values) == 1, (sample, values)


def test_the_implied_cd4_to_cd8_ratio_is_not_credible(b2m):
    """Two bulk samples, one invariant chain, 2.8x apart.

    If this ever drops to something a shared biology could explain, the
    argument in this module's docstring needs revisiting -- and the data might
    then be usable.
    """
    ppm = b2m["abundance_ppm"]
    ratio = ppm["CD4+ T cell"] / ppm["CD8+ T cell"]
    assert ratio > 2.0, (
        f"CD4+/CD8+ B2M is now {ratio:.2f}x; this test exists because 2.79x "
        "between two bulk samples is spectral-counting variation and not a "
        "difference in how visible a T cell is")
    assert b2m["abundance_measured_in"]["CD4+ T cell"] != \
        b2m["abundance_measured_in"]["CD8+ T cell"]


def test_there_is_no_tumour_value_at_all(b2m):
    """Which is the number immune evasion actually turns on."""
    ppm = b2m["abundance_ppm"]
    tumour_ish = [name for name in ppm
                  if any(word in name.lower()
                         for word in ("cancer", "tumour", "tumor", "malignant",
                                      "carcinoma"))]
    assert not tumour_ish, (
        f"a tumour abundance has appeared ({tumour_ish}); the baseline this "
        "column needs may now be sourceable")


def test_the_display_path_exists_and_is_fed_by_invented_numbers():
    """The gap is producer-side, which is why the column stays unwritten."""
    import inspect

    from cognisom.visualization import cell_renderer

    source = inspect.getsource(cell_renderer)
    assert 'color_by == "mhc1"' in source, (
        "the renderer no longer offers an MHC-I view")
    assert "mhc1_expression" in source
    # The demo values encode the biology the renderer expects of a producer.
    assert "0.9 + rng.normal" in source and "0.3 + rng.normal" in source, (
        "the synthetic normal/cancer MHC-I values have moved; if a real "
        "producer landed, this module's docstring is stale")


def test_the_column_is_now_written_from_estimates_not_from_b2m():
    """It is filled, and NOT from the data this file argues against.

    The column was written on 2026-10-22 from labelled estimates, under an
    explicit instruction to accept one if no effect size could be sourced.
    That does not retire anything above: the reason B2M cannot supply the
    baselines is unchanged, and the reason a cancer-versus-normal measurement
    is still missing is unchanged. What changed is that the gap is now filled
    by something that announces itself as a guess instead of being left empty.
    """
    import importlib.util
    from pathlib import Path

    # Loaded by path rather than imported by name: the tests directory is not
    # a package and need not be on sys.path.
    sibling = Path(__file__).with_name(
        "test_which_cell_state_is_actually_simulated.py")
    spec = importlib.util.spec_from_file_location("_cell_state", sibling)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    UNWRITTEN, _column_writers = module.UNWRITTEN, module._column_writers

    assert "mhc1" not in UNWRITTEN
    writers = _column_writers(MHC1)
    assert writers, "mhc1 should have a writer now"

    # And the writer must not be reading B2M abundance.
    from cognisom.core import antigen_presentation

    source = (antigen_presentation.__doc__ or "")
    assert "B2M" in source, (
        "the module that fills the column has to carry the reason it does "
        "not use the obvious data source")
    assert any("B2M" in note for note in antigen_presentation.CAVEATS)
