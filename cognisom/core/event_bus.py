#!/usr/bin/env python3
"""
Event Bus: Inter-Module Communication
======================================

Publish-subscribe system for modules to communicate without tight coupling.
"""

import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class EventTypes:
    """Standard event types"""
    
    # Cellular events
    CELL_DIVIDED = 'cell_divided'
    CELL_DIED = 'cell_died'
    CELL_TRANSFORMED = 'cell_transformed'
    CELL_MIGRATED = 'cell_migrated'
    
    # Molecular events
    EXOSOME_RELEASED = 'exosome_released'
    EXOSOME_UPTAKEN = 'exosome_uptaken'
    MUTATION_OCCURRED = 'mutation_occurred'
    GENE_EXPRESSED = 'gene_expressed'
    
    # Immune events
    IMMUNE_ACTIVATED = 'immune_activated'
    CANCER_KILLED = 'cancer_killed'
    IMMUNE_SUPPRESSED = 'immune_suppressed'
    IMMUNE_RECRUITED = 'immune_recruited'

    # Closed-loop neoantigen presentation (Upgrade 2). Ordered causally:
    # MUTATION_OCCURRED -> PEPTIDE_GENERATED -> PEPTIDE_PRESENTED ->
    # CELL_KILLED_BY_TCELL. End-to-end test asserts this trace appears
    # for a transformed cell whose neoantigen is recognized by the
    # patient's TCR repertoire.
    PEPTIDE_GENERATED = 'peptide_generated'
    PEPTIDE_PRESENTED = 'peptide_presented'
    CELL_KILLED_BY_TCELL = 'cell_killed_by_tcell'

    # TME 4-type classification (Teng et al., Cancer Res 2015).
    # Emitted when ImmuneModule.classify_tme() is called; data carries
    # the full TMEClassification dataclass fields plus the resolved type.
    TME_CLASSIFIED = 'tme_classified'

    # T-cell exhaustion transition (Dolina et al. 2021, Wherry / Bengsch
    # exhaustion literature). Fires once per clone when its encounter
    # counter crosses the exhaustion threshold and it transitions from
    # PD-1-lo precursor to PD-1-hi exhausted. The patent-evidence point
    # is gating ICB rescue on the precursor pool: this event marks
    # clones that have left it.
    TCELL_EXHAUSTED = 'tcell_exhausted'

    # Circadian extrema. CircadianModule's class docstring has always
    # promised these; they were never defined, so nothing could be
    # emitted or subscribed. Emitted once per cell per clock cycle when
    # CLOCK_BMAL1 reaches a local maximum (peak) or minimum (trough).
    CIRCADIAN_PEAK = 'circadian_peak'
    CIRCADIAN_TROUGH = 'circadian_trough'
    
    # Tissue events
    HYPOXIA_DETECTED = 'hypoxia_detected'
    ANGIOGENESIS_STARTED = 'angiogenesis_started'
    METASTASIS_OCCURRED = 'metastasis_occurred'
    VESSEL_FORMED = 'vessel_formed'

    # The causal chain that cognisom/dashboard/_pages/42_patent_pipeline.py
    # reconstructs, HEAD-FIRST (lines ~390, 510, 556-562, 592). EventBus
    # retains these in full, outside the bounded recency log, because the
    # head of the chain is emitted at the start of a run and is therefore
    # exactly what a FIFO recency window evicts first. Evicting the head
    # does not blank the page -- it makes the page report 0 peptides
    # generated and call the mutation HLA-restricted.
    CLOSED_LOOP_TRACE_TYPES = (
        MUTATION_OCCURRED,
        PEPTIDE_GENERATED,
        PEPTIDE_PRESENTED,
        IMMUNE_ACTIVATED,
        TCELL_EXHAUSTED,
        CELL_KILLED_BY_TCELL,
    )


class EventBus:
    """
    Event bus for inter-module communication
    
    Example:
    --------
    # Module A emits event
    event_bus.emit('cell_divided', {'cell_id': 42, 'position': [10, 20, 30]})
    
    # Module B subscribes and receives
    def on_cell_divided(data):
        print(f"Cell {data['cell_id']} divided at {data['position']}")
    
    event_bus.subscribe('cell_divided', on_cell_divided)
    
    # Process events
    event_bus.process_events()  # Calls on_cell_divided
    """
    
    def __init__(self, retain_types: Optional[Sequence[str]] = None):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue: List[tuple] = []

        # (a) Bounded recency window. Debugging aid. FIFO, evicts the
        #     OLDEST entry, so it is the wrong structure for anything
        #     read head-first. Deliberately left at 1000.
        self.event_log: List[tuple] = []
        self.max_log_size = 1000

        # (b) Monotonic accounting, independent of (a). total_processed
        #     used to be reported as len(event_log), which saturates at
        #     max_log_size and under-reports forever after.
        self.total_processed = 0
        self.processed_counts: Counter = Counter()
        self.dropped_from_log = 0

        # (c) Durable trace. The causal chain the patent-evidence page
        #     reconstructs, kept in full and in emission (= causal)
        #     order. When this cap is hit the NEWEST entry is dropped,
        #     not the oldest, because the head is the part that cannot
        #     be re-derived; retained_dropped is then non-zero and the
        #     page must say the trace is incomplete rather than read
        #     absence as biology.
        self.retain_types = set(
            EventTypes.CLOSED_LOOP_TRACE_TYPES
            if retain_types is None else retain_types
        )
        self.retained_log: List[tuple] = []
        self.max_retained_size = 100_000
        self.retained_dropped = 0

        # (d) Cascade guard. A callback may emit while being delivered
        #     and process_events drains that in the same call -- that is
        #     load-bearing for MUTATION_OCCURRED -> PEPTIDE_GENERATED.
        #     What was missing is a floor. Deepest cascade anywhere in
        #     this repo, measured: 1.
        self.max_cascade_depth = 50
        self.cascade_overflows = 0
    
    def emit(self, event_type: str, data: Any = None):
        """
        Emit event (queued for processing)
        
        Parameters:
        -----------
        event_type : str
            Type of event (use EventTypes constants)
        data : any
            Event data (typically dict)
        """
        self.event_queue.append((event_type, data))
    
    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to event type
        
        Parameters:
        -----------
        event_type : str
            Type of event to subscribe to
        callback : callable
            Function to call when event occurs
            Signature: callback(data)
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from event type"""
        if event_type in self.subscribers:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
    
    def process_events(self):
        """
        Process all queued events
        
        Called once per simulation step to deliver events to subscribers.

        Events emitted by a callback are delivered in this same call --
        the closed-loop chain depends on it. They are delivered one
        cascade generation at a time, which is bit-for-bit the same
        order as popping one event at a time (a FIFO drained to
        exhaustion IS breadth-first), and which bounds the loop: a
        callback that re-emits its own trigger is stopped after
        max_cascade_depth generations instead of spinning forever.
        """
        depth = 0
        while self.event_queue:
            if depth > self.max_cascade_depth:
                dropped = len(self.event_queue)
                self.event_queue = []
                self.cascade_overflows += 1
                logger.error(
                    "Event cascade exceeded max_cascade_depth=%d; dropped "
                    "%d pending event(s). A subscriber is re-emitting its "
                    "own trigger. This run's event record is incomplete; "
                    "see cascade_overflows in EventBus.get_statistics().",
                    self.max_cascade_depth, dropped,
                )
                break

            batch = self.event_queue
            self.event_queue = []

            for event_type, data in batch:
                self._record(event_type, data)

                # tuple() so a callback that subscribes or unsubscribes
                # during delivery cannot raise "list changed size during
                # iteration" out of this loop -- that RuntimeError comes
                # from the for statement, outside the try below, and
                # would escape process_events.
                for callback in tuple(self.subscribers.get(event_type, ())):
                    try:
                        callback(data)
                    except Exception:
                        logger.exception(
                            "Event callback %s failed for event %r; the "
                            "event bus continued to the next subscriber. "
                            "The simulation is now missing whatever this "
                            "callback was responsible for.",
                            getattr(callback, "__qualname__", repr(callback)),
                            event_type,
                        )

            depth += 1
    
    def clear_queue(self):
        """Clear event queue"""
        self.event_queue.clear()

    def clear_log(self):
        """Clear the recency window and the retained trace.

        The monotonic counters are NOT reset -- they count every event
        this bus ever processed, which is the whole point of them.
        """
        self.event_log.clear()
        self.retained_log.clear()

    def _record(self, event_type: str, data: Any) -> None:
        """Account for one delivered event.

        Three destinations, deliberately: a monotonic count that never
        saturates, a bounded recency window that evicts oldest-first,
        and a durable trace that evicts newest-first.
        """
        self.total_processed += 1
        self.processed_counts[event_type] += 1

        self.event_log.append((event_type, data))
        if len(self.event_log) > self.max_log_size:
            self.event_log.pop(0)
            self.dropped_from_log += 1

        if event_type in self.retain_types:
            if len(self.retained_log) < self.max_retained_size:
                self.retained_log.append((event_type, data))
            else:
                self.retained_dropped += 1

    def get_trace(self, event_types: Optional[Sequence[str]] = None):
        """Complete causal trace, in emission order, for retained types.

        Unlike get_event_log() this is not a recency window: nothing
        here is evicted by later traffic, so a chain whose head fires at
        t=0 and whose tail fires at t=24h survives intact. Read this,
        not event_log, for anything reconstructed head-first.

        Raises
        ------
        ValueError
            If asked for a type this bus does not retain. Returning a
            silently partial trace is how the bounded log produced
            "0 peptides generated" for a mutation that generated seven.
        """
        if event_types is None:
            return list(self.retained_log)

        wanted = set(event_types)
        missing = wanted - self.retain_types
        if missing:
            raise ValueError(
                f"EventBus does not retain {sorted(missing)}, so a complete "
                f"trace cannot be produced for them. Pass them in "
                f"retain_types when constructing the bus, or read the "
                f"bounded recency window via get_event_log() and accept "
                f"that it may be missing the head of the chain."
            )
        return [(t, d) for t, d in self.retained_log if t in wanted]
    
    def get_event_log(self, event_type: str = None, limit: int = 100):
        """
        Get recent events
        
        Parameters:
        -----------
        event_type : str, optional
            Filter by event type
        limit : int
            Maximum number of events to return
        
        Returns:
        --------
        list : Recent events
        """
        if event_type:
            events = [(t, d) for t, d in self.event_log if t == event_type]
        else:
            events = self.event_log
        
        return events[-limit:]
    
    def get_statistics(self):
        """Get event statistics.

        total_events and event_counts are MONOTONIC: they count every
        event this bus ever processed. They used to be derived from
        event_log, which means they saturated at max_log_size and
        under-reported from then on (1000 reported against 4680
        processed, measured on an 900-cell run). The size of the
        recency window is reported separately as logged_events.
        """
        
        # Counts come from the monotonic Counter, never from the
        # bounded log -- that was the saturation bug.
        
        return {
            'total_events': self.total_processed,
            'logged_events': len(self.event_log),
            'dropped_from_log': self.dropped_from_log,
            'queued_events': len(self.event_queue),
            'event_counts': dict(self.processed_counts),
            'retained_events': len(self.retained_log),
            'retained_dropped': self.retained_dropped,
            'cascade_overflows': self.cascade_overflows,
            'subscribers': {k: len(v) for k, v in self.subscribers.items()},
        }


# Example usage
if __name__ == '__main__':
    print("=" * 60)
    print("Event Bus Test")
    print("=" * 60)
    print()
    
    # Create event bus
    bus = EventBus()
    
    # Subscribe to events
    def on_cell_divided(data):
        print(f"✓ Cell {data['cell_id']} divided at position {data['position']}")
    
    def on_cancer_killed(data):
        print(f"✓ Cancer cell {data['cell_id']} killed by {data['killer_type']}")
    
    bus.subscribe(EventTypes.CELL_DIVIDED, on_cell_divided)
    bus.subscribe(EventTypes.CANCER_KILLED, on_cancer_killed)
    
    # Emit events
    print("Emitting events...")
    bus.emit(EventTypes.CELL_DIVIDED, {'cell_id': 42, 'position': [10, 20, 30]})
    bus.emit(EventTypes.CELL_DIVIDED, {'cell_id': 43, 'position': [15, 25, 35]})
    bus.emit(EventTypes.CANCER_KILLED, {'cell_id': 99, 'killer_type': 'T_cell'})
    print()
    
    # Process events
    print("Processing events...")
    bus.process_events()
    print()
    
    # Statistics
    stats = bus.get_statistics()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 60)
    print("✓ Event bus working!")
    print("=" * 60)
