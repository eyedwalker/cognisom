#!/usr/bin/env python3
"""
Simulation Engine
=================

Master controller that orchestrates all simulation modules.
"""

import time
from typing import Dict, Any, Type
from dataclasses import dataclass

try:
    from .event_bus import EventBus
    from .module_base import SimulationModule
except ImportError:
    from event_bus import EventBus
    from module_base import SimulationModule


@dataclass
class SimulationConfig:
    """Global simulation configuration"""
    
    # Time
    dt: float = 0.01  # hours
    duration: float = 24.0  # hours
    
    # Space
    grid_size: tuple = (200, 200, 100)  # voxels
    resolution: float = 10.0  # μm per voxel
    
    # Performance
    use_gpu: bool = False
    num_threads: int = 4
    
    # Output
    save_interval: float = 1.0  # hours
    output_dir: str = 'output/'
    
    # Modules enabled
    modules_enabled: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.modules_enabled is None:
            self.modules_enabled = {
                'molecular': True,
                'cellular': True,
                'immune': True,
                'vascular': True,
                'lymphatic': True,
                'spatial': True
            }


class SimulationEngine:
    """
    Master simulation engine
    
    Orchestrates all modules, coordinates time stepping,
    routes events, manages state.
    
    Example:
    --------
    engine = SimulationEngine()
    engine.register_module('molecular', MolecularModule)
    engine.register_module('immune', ImmuneModule)
    engine.initialize()
    engine.run(duration=24.0)
    
    state = engine.get_state()
    print(f"Final time: {state['time']}")
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize simulation engine
        
        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration
        """
        self.config = config or SimulationConfig()
        
        # Module registry
        self.modules: Dict[str, SimulationModule] = {}
        
        # Event bus for inter-module communication
        self.event_bus = EventBus()
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.running = False
        self.initialized = False
        
        # Performance tracking
        self.start_time = None
        self.step_times = []
    
    def register_module(self, name: str, module_class: Type[SimulationModule], 
                       config: Dict = None):
        """
        Register a simulation module
        
        Parameters:
        -----------
        name : str
            Module name
        module_class : class
            Module class (inherits from SimulationModule)
        config : dict
            Module-specific configuration
        """
        # Create module instance
        module = module_class(config or {})
        
        # Connect to event bus
        module.set_event_bus(self.event_bus)
        
        # Store module
        self.modules[name] = module
        
        print(f"✓ Registered module: {name}")
    
    def unregister_module(self, name: str):
        """Unregister module"""
        if name in self.modules:
            del self.modules[name]
            print(f"✓ Unregistered module: {name}")
    
    def initialize(self):
        """Initialize all modules"""
        print("\n" + "=" * 60)
        print("Initializing Simulation")
        print("=" * 60)
        print()
        
        for name, module in self.modules.items():
            if module.enabled:
                print(f"Initializing {name}...")
                module.initialize()
        
        self.initialized = True
        print()
        print("✓ All modules initialized")
        print()
    
    def step(self):
        """
        Single time step - updates all modules
        
        Order:
        1. Pre-step (preparation)
        2. Update (main logic)
        3. Post-step (cleanup, events)
        4. Process events (inter-module communication)
        5. Update time
        """
        step_start = time.time()
        
        # 1. Pre-step
        for module in self.modules.values():
            if module.enabled:
                module.pre_step(self.config.dt)
        
        # 2. Update
        for module in self.modules.values():
            if module.enabled:
                module.update(self.config.dt)
        
        # 3. Post-step
        for module in self.modules.values():
            if module.enabled:
                module.post_step(self.config.dt)
        
        # 4. Process events
        self.event_bus.process_events()
        
        # 5. Update time
        self.time += self.config.dt
        self.step_count += 1
        
        # Track performance
        step_time = time.time() - step_start
        self.step_times.append(step_time)
        if len(self.step_times) > 100:
            self.step_times.pop(0)
    
    def run(self, duration: float = None):
        """
        Run simulation for specified duration
        
        Parameters:
        -----------
        duration : float
            Simulation duration in hours (uses config if None)
        """
        if not self.initialized:
            self.initialize()
        
        duration = duration or self.config.duration
        steps = int(duration / self.config.dt)
        
        print("=" * 60)
        print(f"Running Simulation: {duration} hours ({steps} steps)")
        print("=" * 60)
        print()
        
        self.running = True
        self.start_time = time.time()
        
        for step in range(steps):
            if not self.running:
                print("\n⏸ Simulation paused")
                break
            
            self.step()
            
            # Progress update
            if step % 100 == 0 or step == steps - 1:
                progress = (step + 1) / steps * 100
                avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
                print(f"  t={self.time:.2f}h ({step+1}/{steps}, {progress:.1f}%) "
                      f"[{avg_step_time*1000:.2f}ms/step]")
        
        elapsed = time.time() - self.start_time
        print()
        print(f"✓ Simulation complete in {elapsed:.2f}s")
        print()
    
    def pause(self):
        """Pause simulation"""
        self.running = False
    
    def resume(self):
        """Resume simulation"""
        self.running = True
    
    def reset(self):
        """Reset simulation to initial state"""
        print("\n🔄 Resetting simulation...")
        
        self.time = 0.0
        self.step_count = 0
        self.running = False
        
        for module in self.modules.values():
            if module.enabled:
                module.reset()
        
        self.event_bus.clear_queue()
        
        print("✓ Reset complete")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        state = {
            'time': self.time,
            'step_count': self.step_count,
            'running': self.running
        }
        
        # Get state from all modules
        for name, module in self.modules.items():
            state[name] = module.get_state()
        
        return state
    
    def export_to_csv(self, filename: str):
        """Export simulation data to CSV"""
        import csv
        
        state = self.get_state()
        
        # Flatten nested dict
        flat_data = {'time': state['time'], 'step_count': state['step_count']}
        
        for module_name, module_state in state.items():
            if module_name not in ['time', 'step_count', 'running']:
                for key, value in module_state.items():
                    if isinstance(value, (int, float, str)):
                        flat_data[f"{module_name}.{key}"] = value
        
        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_data.keys())
            writer.writeheader()
            writer.writerow(flat_data)
        
        print(f"✓ Data exported to {filename}")
    
    def export_to_json(self, filename: str):
        """Export simulation data to JSON"""
        import json
        
        state = self.get_state()
        
        # Convert numpy types to native Python
        def convert(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        state = convert(state)
        
        # Write to JSON
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✓ Data exported to {filename}")
    
    def export_time_series(self, filename: str, history: list):
        """Export time series data to CSV"""
        import csv
        
        if not history:
            print("⚠ No history data to export")
            return
        
        # Get all keys from first entry
        keys = set()
        for entry in history:
            for module_name, module_state in entry.items():
                if module_name not in ['time', 'step_count', 'running']:
                    for key in module_state.keys():
                        if isinstance(module_state[key], (int, float, str)):
                            keys.add(f"{module_name}.{key}")
        
        keys = ['time', 'step_count'] + sorted(keys)
        
        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            
            for entry in history:
                row = {'time': entry['time'], 'step_count': entry['step_count']}
                
                for module_name, module_state in entry.items():
                    if module_name not in ['time', 'step_count', 'running']:
                        for key, value in module_state.items():
                            if isinstance(value, (int, float, str)):
                                row[f"{module_name}.{key}"] = value
                
                writer.writerow(row)
        
        print(f"✓ Time series exported to {filename} ({len(history)} entries)")
    
    def set_parameter(self, module_name: str, param_name: str, value: Any):
        """
        Change parameter in real-time
        
        Parameters:
        -----------
        module_name : str
            Name of module
        param_name : str
            Parameter name
        value : any
            New value
        """
        if module_name in self.modules:
            self.modules[module_name].set_parameter(param_name, value)
        else:
            print(f"⚠ Module '{module_name}' not found")
    
    def get_parameter(self, module_name: str, param_name: str) -> Any:
        """Get parameter value"""
        if module_name in self.modules:
            return self.modules[module_name].get_parameter(param_name)
        return None
    
    def enable_module(self, name: str):
        """Enable module"""
        if name in self.modules:
            self.modules[name].enable()
            print(f"✓ Enabled module: {name}")
    
    def disable_module(self, name: str):
        """Disable module"""
        if name in self.modules:
            self.modules[name].disable()
            print(f"✓ Disabled module: {name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get simulation statistics
        
        Returns:
        --------
        dict : Statistics
        """
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        return {
            'time': self.time,
            'step_count': self.step_count,
            'modules': len(self.modules),
            'enabled_modules': sum(1 for m in self.modules.values() if m.enabled),
            'avg_step_time': avg_step_time,
            'steps_per_second': 1.0 / avg_step_time if avg_step_time > 0 else 0,
            'event_stats': self.event_bus.get_statistics()
        }


# Test
if __name__ == '__main__':
    from module_base import SimulationModule
    
    # Create test modules
    class TestModuleA(SimulationModule):
        def initialize(self):
            self.counter = 0
            self.subscribe('event_from_b', self.on_event_from_b)
        
        def update(self, dt):
            self.counter += 1
            if self.counter % 50 == 0:
                self.emit_event('event_from_a', {'counter': self.counter})
        
        def get_state(self):
            return {'counter_a': self.counter}
        
        def on_event_from_b(self, data):
            print(f"    Module A received: {data}")
    
    class TestModuleB(SimulationModule):
        def initialize(self):
            self.value = 0
            self.subscribe('event_from_a', self.on_event_from_a)
        
        def update(self, dt):
            self.value += 2
        
        def get_state(self):
            return {'value_b': self.value}
        
        def on_event_from_a(self, data):
            print(f"    Module B received: {data}")
            self.emit_event('event_from_b', {'value': self.value})
    
    print("=" * 60)
    print("Simulation Engine Test")
    print("=" * 60)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    
    # Register modules
    engine.register_module('module_a', TestModuleA)
    engine.register_module('module_b', TestModuleB)
    print()
    
    # Run simulation
    engine.run()
    
    # Get results
    state = engine.get_state()
    print("Final State:")
    for key, value in state.items():
        print(f"  {key}: {value}")
    print()
    
    # Statistics
    stats = engine.get_statistics()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 60)
    print("✓ Simulation engine working!")
    print("=" * 60)
