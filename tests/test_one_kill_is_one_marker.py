"""A kill is drawn once, not twice.

``DEFAULT_EVENT_KINDS`` maps BOTH ``cancer_killed`` and
``cell_killed_by_tcell`` to ``KIND_KILL``, and the comment above it has
always claimed those are "drawn once, not twice". They were not: a
T-cell kill emits both event types, so every writer put two glyphs on
one pixel and every marker count ran ahead of the engine's own.

MEASURED, 4 h Immunotherapy run on the section layout, seed 7:
28 kill-kind events = 19 ``cancer_killed`` + 9 ``cell_killed_by_tcell``,
against an engine ``total_kills`` of 19. The nine duplicated ones are
exactly the T-cell kills.

The recorder still keeps both -- an adaptive kill and an innate one are
two different facts, and only the drawing was ever doubled -- so the
deduplication lives in a function writers call.
"""

from __future__ import annotations

import pytest
from dataclasses import replace

from cognisom.visualization.interaction_recorder import (
    KIND_ACTIVATION, KIND_DEATH, KIND_KILL, VisualEvent, dedupe_kill_events)


def _kill(t, pos, event_type="cancer_killed", cell_id=1):
    return VisualEvent(kind=KIND_KILL, event_type=event_type, time=t,
                       step=int(t * 20), position=pos, cell_id=cell_id)


def test_the_same_kill_reported_twice_is_drawn_once():
    """The real pairing: one T-cell kill, emitted under both names, at
    one instant and one coordinate."""
    pos = (1.0, 2.0, 3.0)
    events = [_kill(4.0, pos, "cancer_killed", cell_id=7),
              _kill(4.0, pos, "cell_killed_by_tcell", cell_id=7)]
    assert len(dedupe_kill_events(events)) == 1


def test_the_surviving_event_is_the_first_one_seen():
    """Order is the record's order, so a writer that draws the result
    keeps the event the recorder saw first rather than an arbitrary
    one."""
    pos = (1.0, 2.0, 3.0)
    first = _kill(4.0, pos, "cancer_killed")
    second = _kill(4.0, pos, "cell_killed_by_tcell")
    assert dedupe_kill_events([first, second]) == [first]


def test_two_kills_at_the_same_instant_in_different_places_both_survive():
    """Two effectors killing simultaneously is ordinary in a dense
    infiltrate. Collapsing them would under-count."""
    events = [_kill(4.0, (1.0, 2.0, 3.0)), _kill(4.0, (9.0, 9.0, 9.0))]
    assert len(dedupe_kill_events(events)) == 2


def test_the_same_killer_at_the_same_place_in_a_later_step_survives():
    """A killer that kills twice from one position, in two different
    steps, made two kills. The key is (time, position) and not a cell
    id for exactly this case."""
    pos = (1.0, 2.0, 3.0)
    events = [_kill(4.0, pos, cell_id=7), _kill(4.5, pos, cell_id=7)]
    assert len(dedupe_kill_events(events)) == 2


def test_non_kill_events_are_never_collapsed():
    """Two activations at one instant and one coordinate are two
    activations. Only the kill map is doubled, so only kills are
    deduplicated."""
    ev = [VisualEvent(kind=KIND_ACTIVATION, event_type="immune_activated",
                      time=1.0, step=20, position=(0.0, 0.0, 0.0)),
          VisualEvent(kind=KIND_ACTIVATION, event_type="immune_activated",
                      time=1.0, step=20, position=(0.0, 0.0, 0.0)),
          VisualEvent(kind=KIND_DEATH, event_type="cell_died", time=1.0,
                      step=20, position=(0.0, 0.0, 0.0))]
    assert len(dedupe_kill_events(ev)) == 3


def test_order_is_preserved_across_kinds():
    """Writers lay markers down in time order; reordering here would
    put a marker on the wrong frame."""
    a = VisualEvent(kind=KIND_ACTIVATION, event_type="immune_activated",
                    time=0.5, step=10, position=(0.0, 0.0, 0.0))
    k = _kill(1.0, (1.0, 1.0, 1.0))
    d = VisualEvent(kind=KIND_DEATH, event_type="cell_died", time=1.5,
                    step=30, position=(2.0, 2.0, 2.0))
    assert dedupe_kill_events([a, k, d]) == [a, k, d]


def test_kills_without_a_position_are_not_merged_into_one():
    """A positionless kill cannot be shown to be the same event as
    another positionless kill at the same instant, and silently
    dropping one would lose a real kill. Only identical (time,
    position) pairs collapse -- and None is only ever equal to None at
    the same instant, which is the genuinely indistinguishable case."""
    events = [_kill(4.0, None), _kill(4.5, None)]
    assert len(dedupe_kill_events(events)) == 2


def test_an_empty_record_is_empty():
    assert dedupe_kill_events([]) == []


@pytest.mark.slow
def test_on_a_real_run_the_markers_match_the_engines_own_count():
    """The claim that matters: after deduplication the number of kill
    markers equals ``total_kills``, the counter the immune module keeps
    for itself. Two independent counts agreeing is the assertion; the
    number they agree on is whatever the run produces, so this does not
    pin a kill count."""
    import random

    import numpy as np

    random.seed(7)
    np.random.seed(7)

    from cognisom.dashboard.engine_runner import EngineRunner

    runner = EngineRunner(dt=0.05, duration=4.0, scenario="Immunotherapy",
                          tissue="section")
    runner.build()
    runner.run()

    raw = list(runner.recorder.events)
    raw_kills = [e for e in raw if e.kind == KIND_KILL]
    deduped_kills = [e for e in dedupe_kill_events(raw) if e.kind == KIND_KILL]
    engine_kills = int(runner.get_time_series()["total_kills"][-1])

    assert len(deduped_kills) == engine_kills, (
        f"{len(deduped_kills)} kill markers against {engine_kills} kills "
        "the engine counted -- re-measure, do not re-pin")
    assert len(raw_kills) >= len(deduped_kills), (
        "deduplication must never invent a marker")


def test_the_detail_of_both_twins_survives_deduplication():
    """Dropping the glyph must not drop the neoantigen.

    The two events of one T-cell kill carry DISJOINT payloads --
    measured 2026-11-06, ``cancer_killed`` carries ``killer_type`` and
    ``cell_killed_by_tcell`` carries ``affinity``, ``hla_allele``,
    ``mutation``, ``peptide`` and ``source_gene``. Keeping only the
    first deleted the evidence for why the kill happened from every
    consumer of the deduplicated list, which is the opposite of what
    this project is for.
    """
    pos = (1.0, 2.0, 3.0)
    innate = _kill(4.0, pos, "cancer_killed")
    adaptive = _kill(4.0, pos, "cell_killed_by_tcell")
    innate = replace(innate, detail={"killer_type": "t_cell"})
    adaptive = replace(adaptive, detail={"peptide": "SYYPTRSAY",
                                         "hla_allele": "HLA-A*01:01"})

    survivors = dedupe_kill_events([innate, adaptive])

    assert len(survivors) == 1
    detail = survivors[0].detail
    assert detail["killer_type"] == "t_cell", "the survivor lost its own detail"
    assert detail["peptide"] == "SYYPTRSAY", (
        "the dropped twin's neoantigen detail was discarded with its glyph")
    assert detail["hla_allele"] == "HLA-A*01:01"


def test_the_survivor_keeps_its_own_value_on_a_collision():
    """A merge must not rewrite the record the recorder saw first."""
    pos = (1.0, 2.0, 3.0)
    first = replace(_kill(4.0, pos, "cancer_killed"), detail={"killer_type": "nk"})
    second = replace(_kill(4.0, pos, "cell_killed_by_tcell"),
                     detail={"killer_type": "t_cell", "peptide": "AAAAAAAAA"})

    survivor = dedupe_kill_events([first, second])[0]

    assert survivor.detail["killer_type"] == "nk"
    assert survivor.detail["peptide"] == "AAAAAAAAA"


@pytest.mark.slow
def test_the_usd_stage_draws_one_cross_per_kill():
    """The geometry a renderer actually consumes, not the writer's report.

    ``author_interaction_markers`` is where the doubling reached the
    RTX/Isaac stage: it was the recorder path that had been fixed, and
    the USD exporter still drew both twins. This counts the distinct
    coordinates inside the authored PointInstancer, so it would fail
    even if the report said the right number.
    """
    pytest.importorskip("pxr")
    import random

    import numpy as np
    from pxr import Usd, UsdGeom

    random.seed(11)
    np.random.seed(11)

    from cognisom.dashboard.engine_runner import EngineRunner
    from cognisom.visualization.run_usd import write_run

    runner = EngineRunner(dt=0.05, duration=4.0, scenario="Immunotherapy",
                          tissue="section")
    runner.build()
    runner.run()

    engine_kills = int(runner.get_time_series()["total_kills"][-1])
    if engine_kills == 0:
        pytest.skip("this run produced no kills, so there is nothing to count")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/kills.usda"
        report = write_run(out, runner)

        assert report.markers["drawn"].get("kill") == engine_kills, (
            f"the writer reports {report.markers['drawn'].get('kill')} kill "
            f"markers against {engine_kills} engine kills")
        assert report.bonds["drawn"] <= engine_kills, (
            "more engagement bonds than kills means the lines doubled too")

        stage = Usd.Stage.Open(out)
        sites = set()
        found = False
        for prim in stage.Traverse():
            if prim.GetTypeName() != "PointInstancer":
                continue
            if "Kill" not in prim.GetName():
                continue
            found = True
            attr = UsdGeom.PointInstancer(prim).GetPositionsAttr()
            samples = attr.GetTimeSamples() or []
            reads = [attr.Get(s) for s in samples] if samples else [attr.Get()]
            for arr in reads:
                for point in (arr or []):
                    sites.add(tuple(round(float(c), 3) for c in point))

        assert found, "no Kill PointInstancer was authored into the stage"
        assert len(sites) == engine_kills, (
            f"{len(sites)} distinct kill coordinates in the stage against "
            f"{engine_kills} kills the engine counted -- re-measure, do not "
            "re-pin")
