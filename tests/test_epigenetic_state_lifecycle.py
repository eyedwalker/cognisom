"""The epigenetic module must stop tracking cells that have died.

It subscribed to CELL_DIVIDED, CELL_TRANSFORMED and HYPOXIA_DETECTED but not
CELL_DIED, so ``cell_epigenetics`` only ever grew. A 6 h Baseline run tracked
90 cells against 8 alive, which means ``avg_methylation`` and
``silenced_genes`` -- both plotted on the dashboard -- were averages over a
population that was mostly dead. The circadian and morphogen modules already
handled the event correctly, so the three disagreed about how many cells
existed.
"""
from __future__ import annotations

import pytest


def _module():
    from cognisom.modules.epigenetic_module import EpigeneticModule

    # Nine of ten modules read .get() off the raw argument rather than
    # self.config, so an explicit dict is required here.
    return EpigeneticModule({})


def test_cell_died_is_subscribed():
    from cognisom.core.event_bus import EventTypes

    mod = _module()
    subscribed: list[str] = []
    mod.subscribe = lambda event, handler: subscribed.append(event)

    mod.initialize()

    assert EventTypes.CELL_DIED in subscribed, (
        "without this subscription the module's cell table only ever grows"
    )


def test_a_death_event_drops_the_cell():
    mod = _module()
    mod.add_cell(1, 'normal')
    mod.add_cell(2, 'cancer')
    assert len(mod.cell_epigenetics) == 2

    mod.on_cell_died({'cell_id': 1})

    assert 1 not in mod.cell_epigenetics
    assert 2 in mod.cell_epigenetics


def test_a_death_event_without_a_cell_id_is_ignored_quietly():
    mod = _module()
    mod.add_cell(1, 'normal')

    mod.on_cell_died({})
    mod.on_cell_died(None)

    assert 1 in mod.cell_epigenetics


def test_removing_an_untracked_cell_is_not_an_error():
    mod = _module()

    mod.on_cell_died({'cell_id': 99})  # never added

    assert mod.cell_epigenetics == {}


def test_tracked_population_matches_the_living_one_through_the_event_bus():
    """End to end: emit deaths on a real bus and check the module keeps up."""
    from cognisom.core.event_bus import EventBus, EventTypes

    bus = EventBus()
    mod = _module()
    mod.event_bus = bus
    mod.initialize()

    for cell_id in range(10):
        mod.add_cell(cell_id, 'normal')
    assert len(mod.cell_epigenetics) == 10

    for cell_id in range(7):
        bus.emit(EventTypes.CELL_DIED, {'cell_id': cell_id})
    bus.process_events()

    assert len(mod.cell_epigenetics) == 3, (
        f"tracked {len(mod.cell_epigenetics)} cells after 7 of 10 died"
    )
    assert set(mod.cell_epigenetics) == {7, 8, 9}
