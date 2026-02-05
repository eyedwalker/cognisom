# 🎮 cognisom Platform - How to Use

## ⚠️ Important Note

The GUI and interactive menus require a **display and keyboard input**, which means:

- ✅ **Works**: When you run them directly in your terminal
- ❌ **Doesn't work**: Through this IDE interface (no interactive input)

---

## ✅ **What DOES Work Here**

### **1. Run Complete Test** (Best option!)
```bash
python3 test_platform.py
```
This runs everything non-interactively and shows you it all works!

### **2. Run Individual Scenarios**
```bash
python3 scenarios/immunotherapy.py
python3 scenarios/hypoxia.py
python3 scenarios/epigenetic_therapy.py
```

### **3. Run Test Scripts**
```bash
python3 test_9_modules.py
python3 test_all_modules.py
```

---

## 🖥️ **To Use Interactive Components**

Open your **actual terminal** (not through IDE) and run:

### **Option 1: Simple Text Menu**
```bash
cd /Users/davidwalker/CascadeProjects/cognisom
python3 simple_menu.py
```

You'll see:
```
======================================================================
🧬 cognisom - Simple Control Menu
======================================================================

1. Run Quick Simulation (0.5 hours)
2. Run Full Simulation (24 hours)
3. View Current State
4. Export Data (CSV)
5. Export Data (JSON)
6. Run Immunotherapy Scenario
7. Run Hypoxia Scenario
8. Generate Report
9. Show Statistics
q. Quit

Enter choice:
```

### **Option 2: GUI Control Panel**
```bash
cd /Users/davidwalker/CascadeProjects/cognisom
python3 ui/control_panel.py
```

A window will open with:
- Control buttons (Play/Pause/Reset)
- Parameter sliders
- Real-time statistics
- Scenario buttons

### **Option 3: Web Dashboard**
```bash
# Terminal 1:
cd /Users/davidwalker/CascadeProjects/cognisom
python3 api/rest_server.py

# Terminal 2:
open web/dashboard.html
```

### **Option 4: Visualization**
```bash
cd /Users/davidwalker/CascadeProjects/cognisom
python3 visualize_complete.py
```

---

## 📊 **Demonstration (Non-Interactive)**

Let me show you what the platform can do right now:

### **Demo 1: Quick Simulation**
```bash
python3 -c "
from core import SimulationEngine, SimulationConfig
from modules import CellularModule, ImmuneModule

engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
engine.register_module('cellular', CellularModule)
engine.register_module('immune', ImmuneModule)
engine.initialize()

immune = engine.modules['immune']
cellular = engine.modules['cellular']
immune.set_cellular_module(cellular)

print('Running 1-hour simulation...')
engine.run()

state = engine.get_state()
print(f'Results:')
print(f'  Time: {state[\"time\"]:.2f}h')
print(f'  Cancer cells: {state[\"cellular\"][\"n_cancer\"]}')
print(f'  Immune kills: {state[\"immune\"][\"total_kills\"]}')
"
```

### **Demo 2: Export Data**
```bash
python3 -c "
from core import SimulationEngine, SimulationConfig
from modules import CellularModule

engine = SimulationEngine(SimulationConfig(dt=0.01, duration=0.5))
engine.register_module('cellular', CellularModule)
engine.initialize()
engine.run()

engine.export_to_csv('demo_results.csv')
engine.export_to_json('demo_results.json')

print('✓ Data exported!')
print('  - demo_results.csv')
print('  - demo_results.json')
"
```

### **Demo 3: Generate Report**
```bash
python3 -c "
from core import SimulationEngine, SimulationConfig
from modules import CellularModule
from api.publisher import Publisher

engine = SimulationEngine(SimulationConfig(dt=0.01, duration=0.5))
engine.register_module('cellular', CellularModule)
engine.initialize()
engine.run()

publisher = Publisher(engine)
files = publisher.generate_all_formats('demo_report')

print('✓ Reports generated!')
for fmt, filename in files.items():
    print(f'  - {filename}')
"
```

---

## 🎯 **Recommended Next Steps**

### **Right Now (Through IDE)**:
```bash
# Run the complete test
python3 test_platform.py
```

### **In Your Terminal** (for interactive use):
```bash
# Open your Mac terminal and run:
cd /Users/davidwalker/CascadeProjects/cognisom

# Try the simple menu:
python3 simple_menu.py

# Or try the GUI:
python3 ui/control_panel.py

# Or try the web dashboard:
python3 api/rest_server.py
# Then open web/dashboard.html in your browser
```

---

## 📝 **Summary**

**What works through IDE**:
- ✅ `python3 test_platform.py` - Complete test
- ✅ `python3 scenarios/*.py` - Run scenarios
- ✅ `python3 test_*.py` - All test scripts
- ✅ Programmatic use (import and use in code)

**What needs terminal**:
- 🖥️ `python3 simple_menu.py` - Interactive menu
- 🖥️ `python3 ui/control_panel.py` - GUI
- 🌐 `python3 api/rest_server.py` - Web server
- 📊 `python3 visualize_complete.py` - Visualization

**The platform is fully functional - it just needs interactive input for menus!**

---

## ✅ **Verify Everything Works**

Run this now:
```bash
python3 test_platform.py
```

This will test all 9 modules, run scenarios, export data, and generate reports - all non-interactively!
