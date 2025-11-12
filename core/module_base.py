#!/usr/bin/env python3
"""
Module Base Class
=================

Base class for all simulation modules. All modules inherit from this
and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class SimulationModule(ABC):
    """
    Base class for all simulation modules
    
    All modules must implement:
    - initialize() - Setup module
    - update(dt) - Update state
    - get_state() - Return current state
    
    Optional methods:
    - pre_step(dt) - Before update
    - post_step(dt) - After update
    - set_parameter(name, value) - Change parameters
    - reset() - Reset to initial state
    
    Example:
    --------
    class MyModule(SimulationModule):
        def initialize(self):
            self.counter = 0
        
        def update(self, dt):
            self.counter += 1
        
        def get_state(self):
            return {'counter': self.counter}
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize module
        
        Parameters:
        -----------
        config : dict
            Module configuration
        """
        self.config = config or {}
        self.event_bus = None
        self.enabled = True
        self.name = self.__class__.__name__
    
    def set_event_bus(self, event_bus):
        """
        Connect to event bus for inter-module communication
        
        Parameters:
        -----------
        event_bus : EventBus
            Event bus instance
        """
        self.event_bus = event_bus
    
    @abstractmethod
    def initialize(self):
        """
        Initialize module (called once at start)
        
        Setup initial state, create objects, etc.
        """
        pass
    
    def pre_step(self, dt: float):
        """
        Pre-step hook (called before update)
        
        Optional. Use for preparation before main update.
        
        Parameters:
        -----------
        dt : float
            Time step in hours
        """
        pass
    
    @abstractmethod
    def update(self, dt: float):
        """
        Update module state
        
        Main simulation logic goes here.
        
        Parameters:
        -----------
        dt : float
            Time step in hours
        """
        pass
    
    def post_step(self, dt: float):
        """
        Post-step hook (called after update)
        
        Optional. Use for cleanup, event emission, etc.
        
        Parameters:
        -----------
        dt : float
            Time step in hours
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Return current state as dictionary
        
        Returns:
        --------
        dict : Current module state
        
        Example:
        --------
        return {
            'n_cells': len(self.cells),
            'time': self.time,
            'statistics': {...}
        }
        """
        pass
    
    def set_parameter(self, name: str, value: Any):
        """
        Change parameter in real-time
        
        Parameters:
        -----------
        name : str
            Parameter name
        value : any
            New value
        """
        if hasattr(self, name):
            setattr(self, name, value)
            print(f"✓ {self.name}.{name} = {value}")
        else:
            print(f"⚠ {self.name} has no parameter '{name}'")
    
    def get_parameter(self, name: str) -> Any:
        """Get parameter value"""
        if hasattr(self, name):
            return getattr(self, name)
        return None
    
    def emit_event(self, event_type: str, data: Any = None):
        """
        Emit event to other modules
        
        Parameters:
        -----------
        event_type : str
            Type of event
        data : any
            Event data
        """
        if self.event_bus:
            self.event_bus.emit(event_type, data)
    
    def subscribe(self, event_type: str, callback):
        """
        Subscribe to events from other modules
        
        Parameters:
        -----------
        event_type : str
            Type of event to subscribe to
        callback : callable
            Function to call when event occurs
        """
        if self.event_bus:
            self.event_bus.subscribe(event_type, callback)
    
    def reset(self):
        """
        Reset module to initial state
        
        Default implementation re-initializes.
        Override if needed.
        """
        self.initialize()
    
    def enable(self):
        """Enable module"""
        self.enabled = True
    
    def disable(self):
        """Disable module"""
        self.enabled = False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get module information
        
        Returns:
        --------
        dict : Module metadata
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'config': self.config
        }


# Example module for testing
class ExampleModule(SimulationModule):
    """Example module implementation"""
    
    def initialize(self):
        """Setup"""
        self.counter = 0
        self.data = []
        
        # Subscribe to events
        self.subscribe('test_event', self.on_test_event)
    
    def update(self, dt):
        """Update"""
        self.counter += 1
        self.data.append(self.counter)
        
        # Emit event every 10 steps
        if self.counter % 10 == 0:
            self.emit_event('milestone', {'counter': self.counter})
    
    def get_state(self):
        """Get state"""
        return {
            'counter': self.counter,
            'data_length': len(self.data)
        }
    
    def on_test_event(self, data):
        """Handle test event"""
        print(f"  ExampleModule received: {data}")


# Test
if __name__ == '__main__':
    from event_bus import EventBus
    
    print("=" * 60)
    print("Module Base Class Test")
    print("=" * 60)
    print()
    
    # Create event bus
    bus = EventBus()
    
    # Create module
    module = ExampleModule({'test_param': 42})
    module.set_event_bus(bus)
    
    # Initialize
    print("Initializing module...")
    module.initialize()
    print(f"  Initial state: {module.get_state()}")
    print()
    
    # Subscribe to milestone events
    def on_milestone(data):
        print(f"  ✓ Milestone reached: {data['counter']}")
    
    bus.subscribe('milestone', on_milestone)
    
    # Run simulation
    print("Running 25 steps...")
    for step in range(25):
        module.update(dt=0.01)
        bus.process_events()
    
    print()
    print(f"Final state: {module.get_state()}")
    print()
    
    # Test parameter setting
    print("Testing parameter setting...")
    module.set_parameter('counter', 100)
    print(f"  State after change: {module.get_state()}")
    print()
    
    # Test reset
    print("Testing reset...")
    module.reset()
    print(f"  State after reset: {module.get_state()}")
    print()
    
    print("=" * 60)
    print("✓ Module base class working!")
    print("=" * 60)
