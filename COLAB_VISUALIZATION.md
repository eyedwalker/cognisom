# 📊 Visualization on Google Colab

## ⚠️ What Doesn't Work on Colab

### **GUI Control Panel** (`ui/control_panel.py`)
- ❌ Requires tkinter (desktop GUI)
- ❌ Needs display window
- ❌ Won't work on Colab

### **3D Visualization** (`visualize_complete.py`)
- ❌ Uses matplotlib interactive mode
- ❌ Needs display window
- ❌ Won't work on Colab

---

## ✅ What DOES Work on Colab

### **1. Static Plots** ✅
### **2. Inline Visualizations** ✅
### **3. Data Export** ✅
### **4. HTML Reports** ✅
### **5. Web Dashboard** (with ngrok) ✅

---

## 🎨 Colab-Friendly Visualizations

### **Cell 1: Setup**
```python
!git clone https://github.com/eyedwalker/cognisom.git
%cd cognisom
!pip install -q numpy scipy matplotlib flask flask-cors pandas

import matplotlib.pyplot as plt
import numpy as np
from core import SimulationEngine, SimulationConfig
from modules import *

print("✅ Setup complete!")
```

---

### **Cell 2: Run Simulation with Tracking**

```python
# Create simulation
engine = SimulationEngine(SimulationConfig(dt=0.01, duration=24.0))

engine.register_module('cellular', CellularModule, {
    'n_normal_cells': 50,
    'n_cancer_cells': 10
})
engine.register_module('immune', ImmuneModule, {
    'n_t_cells': 15,
    'n_nk_cells': 10
})
engine.register_module('vascular', VascularModule)

engine.initialize()

# Link modules
immune = engine.modules['immune']
cellular = engine.modules['cellular']
vascular = engine.modules['vascular']

immune.set_cellular_module(cellular)
vascular.set_cellular_module(cellular)

# Track data over time
time_points = []
cancer_counts = []
normal_counts = []
immune_kills = []
oxygen_levels = []

print("🚀 Running 24-hour simulation...")

# Run with tracking
for step in range(2400):  # 24 hours
    engine.step()
    
    # Record every hour
    if step % 100 == 0:
        state = engine.get_state()
        time_points.append(state['time'])
        cancer_counts.append(state['cellular']['n_cancer'])
        normal_counts.append(state['cellular']['n_normal'])
        immune_kills.append(state['immune']['total_kills'])
        oxygen_levels.append(state['vascular']['avg_cell_O2'])

print("✅ Simulation complete!")
```

---

### **Cell 3: Plot Cell Populations**

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(time_points, cancer_counts, 'r-', linewidth=2, label='Cancer Cells')
plt.plot(time_points, normal_counts, 'g-', linewidth=2, label='Normal Cells')
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Cell Count', fontsize=12)
plt.title('Cell Population Over Time', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(time_points, immune_kills, 'b-', linewidth=2)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Total Kills', fontsize=12)
plt.title('Immune System Activity', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"📊 Final Results:")
print(f"  Cancer cells: {cancer_counts[-1]}")
print(f"  Normal cells: {normal_counts[-1]}")
print(f"  Immune kills: {immune_kills[-1]}")
```

---

### **Cell 4: Plot Oxygen Levels**

```python
plt.figure(figsize=(10, 5))
plt.plot(time_points, oxygen_levels, 'c-', linewidth=2)
plt.axhline(y=0.05, color='r', linestyle='--', label='Hypoxic Threshold')
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Oxygen Level', fontsize=12)
plt.title('Average Cellular Oxygen Over Time', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"📊 Oxygen Stats:")
print(f"  Average: {np.mean(oxygen_levels):.3f}")
print(f"  Min: {np.min(oxygen_levels):.3f}")
print(f"  Max: {np.max(oxygen_levels):.3f}")
```

---

### **Cell 5: Cell Spatial Distribution**

```python
# Get current cell positions
cellular = engine.modules['cellular']

cancer_positions = []
normal_positions = []

for cell_id, cell in cellular.cells.items():
    if cell.cell_type == 'cancer':
        cancer_positions.append(cell.position)
    else:
        normal_positions.append(cell.position)

# Convert to arrays
if cancer_positions:
    cancer_pos = np.array(cancer_positions)
else:
    cancer_pos = np.array([]).reshape(0, 3)

if normal_positions:
    normal_pos = np.array(normal_positions)
else:
    normal_pos = np.array([]).reshape(0, 3)

# Plot 2D projection (XY plane)
plt.figure(figsize=(10, 10))
if len(normal_pos) > 0:
    plt.scatter(normal_pos[:, 0], normal_pos[:, 1], 
                c='green', s=100, alpha=0.6, label='Normal Cells')
if len(cancer_pos) > 0:
    plt.scatter(cancer_pos[:, 0], cancer_pos[:, 1], 
                c='red', s=150, alpha=0.8, label='Cancer Cells', marker='*')

plt.xlabel('X Position (μm)', fontsize=12)
plt.ylabel('Y Position (μm)', fontsize=12)
plt.title('Cell Spatial Distribution (2D View)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

print(f"📍 Spatial Stats:")
print(f"  Cancer cells: {len(cancer_pos)}")
print(f"  Normal cells: {len(normal_pos)}")
```

---

### **Cell 6: Comparison Plot (Multiple Scenarios)**

```python
# Run 3 different scenarios
scenarios = {
    'Baseline': {'n_t_cells': 10},
    'Boosted Immune': {'n_t_cells': 30},
    'Suppressed Immune': {'n_t_cells': 5}
}

results = {}

for name, params in scenarios.items():
    print(f"Running {name}...")
    
    # Create engine
    eng = SimulationEngine(SimulationConfig(dt=0.01, duration=24.0))
    eng.register_module('cellular', CellularModule, {'n_cancer_cells': 10})
    eng.register_module('immune', ImmuneModule, params)
    eng.initialize()
    
    # Link
    imm = eng.modules['immune']
    cell = eng.modules['cellular']
    imm.set_cellular_module(cell)
    
    # Track
    times = []
    cancers = []
    
    for step in range(2400):
        eng.step()
        if step % 100 == 0:
            state = eng.get_state()
            times.append(state['time'])
            cancers.append(state['cellular']['n_cancer'])
    
    results[name] = (times, cancers)

# Plot comparison
plt.figure(figsize=(12, 6))
colors = ['blue', 'green', 'red']
for (name, (times, cancers)), color in zip(results.items(), colors):
    plt.plot(times, cancers, linewidth=2, label=name, color=color)

plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Cancer Cell Count', fontsize=12)
plt.title('Immunotherapy Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

print("✅ Comparison complete!")
for name, (times, cancers) in results.items():
    print(f"  {name}: {cancers[-1]} cancer cells")
```

---

### **Cell 7: Heatmap Visualization**

```python
# Create a grid-based heatmap
grid_size = 20
cell_density = np.zeros((grid_size, grid_size))

# Count cells in each grid square
for cell_id, cell in cellular.cells.items():
    x, y, z = cell.position
    # Map to grid coordinates
    grid_x = int((x + 500) / 1000 * grid_size)
    grid_y = int((y + 500) / 1000 * grid_size)
    
    # Clamp to grid
    grid_x = max(0, min(grid_size - 1, grid_x))
    grid_y = max(0, min(grid_size - 1, grid_y))
    
    cell_density[grid_y, grid_x] += 1

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(cell_density, cmap='hot', interpolation='nearest')
plt.colorbar(label='Cell Count')
plt.xlabel('X Grid', fontsize=12)
plt.ylabel('Y Grid', fontsize=12)
plt.title('Cell Density Heatmap', fontsize=14, fontweight='bold')
plt.show()

print(f"📊 Density Stats:")
print(f"  Max density: {np.max(cell_density):.0f} cells/region")
print(f"  Avg density: {np.mean(cell_density):.1f} cells/region")
```

---

### **Cell 8: Generate Interactive HTML Report**

```python
from api.publisher import Publisher

# Generate report
publisher = Publisher(engine)
files = publisher.generate_all_formats('colab_report')

print("✅ Reports generated:")
for fmt, filename in files.items():
    print(f"  - {filename}")

# Display HTML report inline
from IPython.display import HTML, display
with open('colab_report.html', 'r') as f:
    display(HTML(f.read()))
```

---

### **Cell 9: Download All Results**

```python
from google.colab import files

# Export data
engine.export_to_csv('simulation_data.csv')
engine.export_to_json('simulation_data.json')

# Download files
files.download('simulation_data.csv')
files.download('simulation_data.json')
files.download('colab_report.html')

print("✅ Files downloaded!")
```

---

## 🌐 Web Dashboard on Colab (Advanced)

### **Cell 10: Run Web Dashboard with ngrok**

```python
# Install ngrok
!pip install -q pyngrok

from pyngrok import ngrok
import threading

# Start API server in background
def run_api():
    import sys
    sys.path.insert(0, '/content/cognisom')
    from api.rest_server import app
    app.run(host='0.0.0.0', port=5000)

thread = threading.Thread(target=run_api, daemon=True)
thread.start()

# Wait for server to start
import time
time.sleep(3)

# Create public URL
public_url = ngrok.connect(5000)
print(f"🌐 API Server running at: {public_url}")
print(f"📊 Web Dashboard: {public_url}/dashboard")
print("\nClick the link above to access the web interface!")
```

---

## 📋 Summary

### **What Works on Colab**:
- ✅ Static plots (matplotlib)
- ✅ Inline visualizations
- ✅ Data export
- ✅ HTML reports
- ✅ Web dashboard (with ngrok)

### **What Doesn't Work**:
- ❌ GUI control panel (tkinter)
- ❌ Interactive 3D plots
- ❌ Desktop windows

### **Best for Colab**:
1. Run simulations
2. Generate plots
3. Export data
4. Create reports
5. Download results

---

## 🎯 Recommended Workflow

**On Colab** (Testing & Analysis):
1. Run simulations
2. Generate plots
3. Export data
4. Download results

**On Local Machine** (Interactive):
1. GUI control panel
2. 3D visualization
3. Real-time interaction

---

**Colab is perfect for running simulations and generating plots, but not for interactive GUIs!** 📊✨
