#!/usr/bin/env python3
"""
Event Bus: Inter-Module Communication
======================================

Publish-subscribe system for modules to communicate without tight coupling.
"""

from typing import Callable, Dict, List, Any


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
    
    # Tissue events
    HYPOXIA_DETECTED = 'hypoxia_detected'
    ANGIOGENESIS_STARTED = 'angiogenesis_started'
    METASTASIS_OCCURRED = 'metastasis_occurred'
    VESSEL_FORMED = 'vessel_formed'


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
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue: List[tuple] = []
        self.event_log: List[tuple] = []
        self.max_log_size = 1000
    
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
        """
        while self.event_queue:
            event_type, data = self.event_queue.pop(0)
            
            # Log event
            self.event_log.append((event_type, data))
            if len(self.event_log) > self.max_log_size:
                self.event_log.pop(0)
            
            # Deliver to subscribers
            if event_type in self.subscribers:
                for callback in self.subscribers[event_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        print(f"Error in event callback for {event_type}: {e}")
    
    def clear_queue(self):
        """Clear event queue"""
        self.event_queue.clear()
    
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
        """Get event statistics"""
        from collections import Counter
        
        event_types = [event_type for event_type, _ in self.event_log]
        counts = Counter(event_types)
        
        return {
            'total_events': len(self.event_log),
            'queued_events': len(self.event_queue),
            'event_counts': dict(counts),
            'subscribers': {k: len(v) for k, v in self.subscribers.items()}
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
