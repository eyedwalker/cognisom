# 🏗️ cognisom Integration Plan

## Problem: Independent Demos → Unified Platform

**Current**: 4 standalone demos, no integration
**Target**: Unified platform with menu system, real-time control, API access

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │   CLI    │  │   GUI    │  │   Web    │  │   API   ││
│  │   Menu   │  │  Panel   │  │Dashboard │  │  REST   ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘│
└───────┼─────────────┼─────────────┼─────────────┼─────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │    SimulationEngine (Master)      │
        │  - Orchestrates all modules       │
        │  - Event bus communication        │
        │  - Time stepping                  │
        │  - State management               │
        └─────────────────┬─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │         Module Registry            │
        │  ┌──────────┐  ┌──────────┐       │
        │  │Molecular │  │ Cellular │       │
        │  └──────────┘  └──────────┘       │
        │  ┌──────────┐  ┌──────────┐       │
        │  │  Immune  │  │ Vascular │       │
        │  └──────────┘  └──────────┘       │
        │  ┌──────────┐  ┌──────────┐       │
        │  │Lymphatic │  │ Spatial  │       │
        │  └──────────┘  └──────────┘       │
        └───────────────────────────────────┘
```

---

## Key Components

### 1. **SimulationEngine** (Master Controller)
```python
- Loads/unloads modules
- Coordinates time stepping
- Routes events between modules
- Manages configuration
- Handles I/O
```

### 2. **EventBus** (Inter-Module Communication)
```python
# Module A emits
event_bus.emit('cell_divided', {'cell_id': 42})

# Module B receives
event_bus.subscribe('cell_divided', callback)
```

### 3. **Module System** (Pluggable Components)
```python
class SimulationModule:
    def initialize()
    def update(dt)
    def get_state()
    def set_parameter(name, value)
```

### 4. **Menu System** (User Interface)
```
Main Menu:
1. Run Simulation
2. Configure Settings
3. View Results
4. Run Scenario
5. API Mode
```

---

## Implementation Steps

### **Week 1: Core Architecture**
```bash
# Create structure
mkdir -p core modules ui api scenarios

# Files to create:
core/simulation_engine.py    # Master controller
core/event_bus.py             # Communication
core/module_base.py           # Base class
modules/molecular_module.py   # Refactored
```

### **Week 2: Menu System**
```bash
# Files to create:
ui/menu_system.py            # CLI menu
ui/settings_panel.py         # Configuration
cognisom_app.py              # Main entry point
```

### **Week 3: Integration**
```bash
# Refactor existing demos into modules
modules/cellular_module.py
modules/immune_module.py
modules/vascular_module.py
modules/lymphatic_module.py
modules/spatial_module.py
```

### **Week 4: GUI & API**
```bash
# Add advanced interfaces
ui/control_panel.py          # tkinter GUI
api/rest_api.py              # Flask API
api/websocket_server.py      # Real-time updates
```

---

## Usage Examples

### **CLI Menu**
```bash
$ python3 cognisom_app.py

cognisom: Multi-Scale Cellular Simulation
==========================================
1. Run Simulation
2. Configure Settings
3. View Results
4. Run Scenario
5. API Mode
q. Quit

Choice: 2

Settings
========
Modules:
  ✓ molecular
  ✓ cellular
  ✓ immune
  ✓ vascular
  
Time: dt=0.01h, duration=24h
Space: 200x200x100, 10μm/voxel

1. Toggle Module
2. Change Parameters
...
```

### **Programmatic**
```python
from cognisom import SimulationEngine

engine = SimulationEngine()
engine.register_module('molecular', MolecularModule)
engine.register_module('immune', ImmuneModule)
engine.initialize()
engine.run(duration=24.0)

state = engine.get_state()
```

### **REST API**
```bash
# Start simulation
curl -X POST http://localhost:5000/api/simulation/start

# Get state
curl http://localhost:5000/api/simulation/state

# Change parameter
curl -X POST http://localhost:5000/api/simulation/parameter \
  -d '{"module": "immune", "param": "n_cells", "value": 50}'
```

---

## Benefits

✅ **Modularity** - Add/remove modules easily
✅ **Flexibility** - Multiple interfaces (CLI/GUI/API)
✅ **Scalability** - Modules can be distributed
✅ **User-Friendly** - Menu for beginners, API for experts
✅ **Extensibility** - Plugin system for new modules

---

## File Structure

```
cognisom/
├── cognisom_app.py              # Main entry
├── core/
│   ├── simulation_engine.py     # Master
│   ├── event_bus.py             # Events
│   └── module_base.py           # Base
├── modules/
│   ├── molecular_module.py
│   ├── cellular_module.py
│   ├── immune_module.py
│   ├── vascular_module.py
│   ├── lymphatic_module.py
│   └── spatial_module.py
├── ui/
│   ├── menu_system.py
│   ├── control_panel.py
│   └── visualization_hub.py
└── api/
    ├── rest_api.py
    └── websocket_server.py
```

---

## Next: Start Implementation

Create core architecture first, then refactor existing code into modules.
