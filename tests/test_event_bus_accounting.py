import logging
import pytest
from cognisom.core.event_bus import EventBus, EventTypes


def test_total_events_does_not_saturate_at_the_log_bound():
    bus = EventBus()
    for i in range(2000):
        bus.emit('filler', {'i': i})
    bus.process_events()
    s = bus.get_statistics()
    assert s['total_events'] == 2000
    assert s['logged_events'] == 1000
    assert s['dropped_from_log'] == 1000
    assert len(bus.event_log) == 1000
    assert sum(s['event_counts'].values()) == 2000


def test_closed_loop_trace_survives_a_log_overflow():
    bus = EventBus()
    bus.emit(EventTypes.MUTATION_OCCURRED, {'cell_id': 1})
    bus.emit(EventTypes.PEPTIDE_GENERATED, {'cell_id': 1, 'n_peptides': 7})
    bus.emit(EventTypes.PEPTIDE_PRESENTED, {'cell_id': 1})
    for i in range(2000):
        bus.emit(EventTypes.GENE_EXPRESSED, {'i': i})
    bus.emit(EventTypes.IMMUNE_ACTIVATED, {'cell_id': 1})
    bus.emit(EventTypes.TCELL_EXHAUSTED, {'cell_id': 1})
    bus.emit(EventTypes.CELL_KILLED_BY_TCELL, {'cell_id': 1})
    bus.process_events()

    assert [t for t, _ in bus.get_trace()] == list(EventTypes.CLOSED_LOOP_TRACE_TYPES)

    def page_lookup(log, cancer_id=1):
        return next(
            (data.get("n_peptides", 0)
             for evt, data in log
             if evt == EventTypes.PEPTIDE_GENERATED
             and isinstance(data, dict) and data.get("cell_id") == cancer_id),
            0,
        )
    assert page_lookup(bus.get_trace()) == 7
    assert bus.get_statistics()['total_events'] == 2006

    # negative control
    assert page_lookup(bus.event_log) == 0
    assert EventTypes.MUTATION_OCCURRED not in [t for t, _ in bus.event_log]
    print("window retains:", sorted(set(t for t, _ in bus.event_log)))


def test_get_trace_refuses_a_type_it_does_not_retain():
    bus = EventBus()
    with pytest.raises(ValueError, match='cell_divided'):
        bus.get_trace([EventTypes.CELL_DIVIDED])


def test_retained_trace_drops_the_newest_not_the_head():
    bus = EventBus()
    bus.max_retained_size = 3
    for i in range(5):
        bus.emit(EventTypes.PEPTIDE_PRESENTED, {'i': i})
    bus.process_events()
    assert [d['i'] for _, d in bus.get_trace()] == [0, 1, 2]
    assert bus.get_statistics()['retained_dropped'] == 2


def test_self_reemitting_callback_terminates(caplog):
    bus = EventBus()
    calls = []

    def loop(data):
        calls.append(1)
        bus.emit('loop', data)
    bus.subscribe('loop', loop)
    bus.emit('loop', {})
    with caplog.at_level(logging.ERROR, logger='cognisom.core.event_bus'):
        bus.process_events()
    assert len(calls) == bus.max_cascade_depth + 1 == 51
    assert bus.get_statistics()['cascade_overflows'] == 1
    errs = [r for r in caplog.records if r.name == 'cognisom.core.event_bus']
    assert len(errs) == 1 and 'max_cascade_depth' in errs[0].getMessage()


def test_a_legitimate_cascade_is_still_delivered_in_one_call():
    bus = EventBus()
    bus.subscribe('a', lambda d: bus.emit('b', d))
    bus.subscribe('b', lambda d: bus.emit('c', d))
    bus.emit('a', {})
    bus.process_events()
    assert [t for t, _ in bus.event_log] == ['a', 'b', 'c']
    assert bus.get_statistics()['cascade_overflows'] == 0
    assert bus.event_queue == []


def test_breadth_first_order_is_unchanged():
    bus = EventBus()
    bus.subscribe('A', lambda d: bus.emit('A1', d))
    bus.subscribe('B', lambda d: bus.emit('B1', d))
    bus.emit('A', {})
    bus.emit('B', {})
    bus.process_events()
    assert [t for t, _ in bus.event_log] == ['A', 'B', 'A1', 'B1']


def test_callback_exception_goes_to_logging_not_stdout(caplog, capsys):
    bus = EventBus()
    ran = []

    def broken(data):
        raise ValueError('broken')

    def good(data):
        ran.append(1)
    bus.subscribe('x', broken)
    bus.subscribe('x', good)
    bus.emit('x', {})
    with caplog.at_level(logging.ERROR, logger='cognisom.core.event_bus'):
        bus.process_events()
    out = capsys.readouterr().out
    assert out == '', repr(out)
    assert 'Error in event callback' not in out
    recs = [r for r in caplog.records if r.name == 'cognisom.core.event_bus']
    assert len(recs) == 1
    r = recs[0]
    assert r.levelname == 'ERROR'
    assert r.exc_info is not None and r.exc_info[0] is ValueError
    assert 'broken' in (r.exc_text or '')
    msg = r.getMessage()
    assert 'broken' in msg and 'x' in msg
    assert ran == [1]


def test_subscriber_list_mutated_during_delivery_does_not_escape():
    bus = EventBus()

    def selfremove(data):
        bus.unsubscribe('y', selfremove)
    bus.subscribe('y', selfremove)
    bus.subscribe('y', lambda d: None)
    bus.emit('y', {})
    bus.process_events()
    assert bus.subscribers['y'] != []


def test_an_engine_reset_clears_the_retained_trace():
    """A reset-and-rerun engine must not show the previous run's trace.

    The retained closed-loop log deliberately survives the 1,000-entry
    recency window, which is the whole point of it -- but it also survived
    ``SimulationEngine.reset()``, because reset cleared only the queue. Page
    42 then read a MUTATION_OCCURRED belonging to a run that had been thrown
    away, and reported it as this run's causal chain.

    The monotonic counters intentionally do NOT reset: they are lifetime
    counts of everything the bus processed, which is what makes them a
    trustworthy denominator.
    """
    from cognisom.core.event_bus import EventBus, EventTypes

    bus = EventBus()

    bus.emit(EventTypes.MUTATION_OCCURRED, {"cell_id": 1})
    bus.emit(EventTypes.CELL_KILLED_BY_TCELL, {"cell_id": 1})
    bus.process_events()
    assert len(bus.get_trace()) == 2

    bus.clear_log()

    bus.emit(EventTypes.MUTATION_OCCURRED, {"cell_id": 2})
    bus.process_events()

    trace = bus.get_trace()
    mutations = [data for evt, data in trace if evt == EventTypes.MUTATION_OCCURRED]
    assert len(mutations) == 1, (
        f"the discarded run's events are still in the trace: {trace}"
    )
    assert mutations[0]["cell_id"] == 2, "the trace shows the first run's cell"

    # Lifetime accounting survives the reset.
    assert bus.get_statistics()["total_events"] == 3


def test_the_engine_wires_clear_log_into_reset():
    """Pin the wiring, not just the bus method.

    clear_log() existing is not the fix; reset() calling it is.
    """
    import inspect

    from cognisom.core.simulation_engine import SimulationEngine

    source = inspect.getsource(SimulationEngine.reset)
    assert "clear_log" in source, (
        "SimulationEngine.reset clears the queue but not the retained log, "
        "so a rerun shows the previous run's causal trace"
    )
