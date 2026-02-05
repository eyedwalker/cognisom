# 🚀 Getting Started with cognisom

## ✅ Platform Status: WORKING!

All tests passed! The platform is ready to use.

---

## 🎯 Quick Start Options

### **Option 1: Run Complete Test** (Recommended First!)

```bash
python3 test_platform.py
```

**This will**:
- Test all 9 modules
- Run a simulation
- Export data
- Run a scenario
- Generate reports
- Verify everything works

**Expected output**: All tests pass ✓

---

### **Option 2: Run Scenarios** (Easy Research)

```bash
# Immunotherapy scenario
python3 scenarios/immunotherapy.py

# Hypoxia scenario
python3 scenarios/hypoxia.py

# Epigenetic therapy
python3 scenarios/epigenetic_therapy.py

# Chronotherapy
python3 scenarios/chronotherapy.py

# Circadian disruption
python3 scenarios/circadian_disruption.py
```

---

### **Option 3: Web Dashboard** (Best for Demos)

```bash
# Terminal 1: Start API server
python3 api/rest_server.py

# Terminal 2: Open dashboard
open web/dashboard.html
# Or manually open web/dashboard.html in your browser

# Use the web interface to:
# - Start/stop simulation
# - View real-time statistics
# - Run scenarios
# - Export data
```

---

### **Option 4: GUI Control Panel** (Interactive Desktop)

```bash
python3 ui/control_panel.py

# Features:
# - Real-time parameter sliders
# - Play/Pause/Reset buttons
# - Live statistics
# - Scenario selection
```

**Note**: Requires display (won't work over SSH without X11)

---

### **Option 5: Complete Visualization** (See Everything)

```bash
python3 visualize_complete.py

# Shows 9 panels:
# - 3D tissue view
# - Statistics
# - Oxygen gradient
# - Circadian rhythms
# - Morphogen gradients
# - Cell fates
# - Epigenetic states
# - Cell population
# - Immune activity
```

**Note**: Requires display

---

### **Option 6: Programmatic Use** (For Your Code)

```python
from core import SimulationEngine, SimulationConfig
from modules import CellularModule, ImmuneModule

# Create engine
engine = SimulationEngine(SimulationConfig(dt=0.01, duration=24.0))

# Register modules
engine.register_module('cellular', CellularModule, {
    'n_normal_cells': 20,
    'n_cancer_cells': 5
})
engine.register_module('immune', ImmuneModule, {
    'n_t_cells': 10,
    'n_nk_cells': 5
})

# Initialize
engine.initialize()

# Link modules
immune = engine.modules['immune']
cellular = engine.modules['cellular']
immune.set_cellular_module(cellular)

# Run
engine.run()

# Get results
state = engine.get_state()
print(f"Cancer cells: {state['cellular']['n_cancer']}")
print(f"Immune kills: {state['immune']['total_kills']}")

# Export
engine.export_to_csv('results.csv')
engine.export_to_json('results.json')
```

---

## 📊 What Each Component Does

### **test_platform.py**
- Comprehensive test of all features
- Non-interactive (good for CI/CD)
- Verifies everything works
- **Run this first!**

### **scenarios/*.py**
- Pre-built research experiments
- Immunotherapy, hypoxia, epigenetic therapy, etc.
- Complete with analysis
- **Good for quick research**

### **api/rest_server.py**
- REST API server (Flask)
- Enables web dashboard
- Remote control
- **Good for web access**

### **web/dashboard.html**
- Browser-based interface
- Real-time statistics
- Control buttons
- **Good for demos**

### **ui/control_panel.py**
- Desktop GUI (tkinter)
- Parameter sliders
- Live updates
- **Good for exploration**

### **visualize_complete.py**
- 9-panel visualization
- Real-time 3D view
- All modules visible
- **Good for understanding**

---

## 🎓 Recommended Learning Path

### **Day 1: Verify Everything Works**
```bash
python3 test_platform.py
```

### **Day 2: Run Scenarios**
```bash
python3 scenarios/immunotherapy.py
python3 scenarios/hypoxia.py
```

### **Day 3: Try Web Dashboard**
```bash
python3 api/rest_server.py
# Then open web/dashboard.html
```

### **Day 4: Programmatic Use**
Create your own script using the API

### **Day 5: Visualization**
```bash
python3 visualize_complete.py
```

---

## 🔧 Troubleshooting

### **"Module not found" error**
```bash
# Make sure you're in the cognisom directory
cd /Users/davidwalker/CascadeProjects/cognisom
```

### **GUI won't start**
- Requires display (won't work over SSH)
- Try web dashboard instead

### **API server won't start**
```bash
# Install Flask if needed
pip install flask flask-cors
```

### **Visualization won't show**
- Requires display
- Check matplotlib backend
- Try non-interactive mode

---

## 📝 Next Steps

### **For Research**:
1. Run scenarios
2. Modify parameters
3. Export data
4. Analyze results

### **For Development**:
1. Create new scenarios
2. Add new modules
3. Extend functionality
4. Integrate with your code

### **For Publication**:
1. Run validation studies
2. Generate reports
3. Create figures
4. Write paper

---

## 💡 Tips

**Best for quick tests**: `python3 test_platform.py`

**Best for research**: Scenarios (`python3 scenarios/immunotherapy.py`)

**Best for demos**: Web dashboard (`python3 api/rest_server.py` + open `web/dashboard.html`)

**Best for exploration**: GUI (`python3 ui/control_panel.py`)

**Best for understanding**: Visualization (`python3 visualize_complete.py`)

**Best for integration**: Programmatic API (import and use in your code)

---

## ✅ Verification Checklist

- [x] Platform test passes
- [x] Scenarios run
- [x] Data exports
- [x] Reports generate
- [x] All 9 modules work
- [x] Performance optimizations active

**Status**: READY TO USE! 🎉

---

## 📧 Need Help?

Check documentation:
- `PLATFORM_COMPLETE.md` - Full overview
- `QUICK_WINS_COMPLETE.md` - GUI, scenarios, export
- `README_PLATFORM.md` - Complete guide

Or run:
```bash
python3 test_platform.py  # Verify everything works
```

---

**You're ready to go! Start with `python3 test_platform.py` to verify everything works, then choose your preferred interface!** 🚀
