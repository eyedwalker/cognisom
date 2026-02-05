# 🎬 Live Interactive Cellular Visualization

## Overview

Real-time, animated visualization showing **both internal and external** cellular dynamics!

**What you'll see**:
- ✅ Multiple cells interacting in 3D space
- ✅ Internal cell structure (DNA, RNA, proteins, organelles)
- ✅ Animated organelles (mitochondria rotate!)
- ✅ mRNA transport (nucleus → cytoplasm)
- ✅ Cell-cell interactions and signaling
- ✅ Environmental gradients (oxygen diffusion)
- ✅ Real-time molecular counts

---

## 🚀 Quick Start

### Run the Live Demo

```bash
python3 live_demo.py
```

**That's it!** A window will open with 5 panels showing live simulation.

---

## 📊 What Each Panel Shows

### **Top Left: Spatial Environment**
- Multiple cells (5 cells)
- Cell positions and interactions
- Oxygen gradient (blue background)
- Red dashed lines = cell-cell signaling
- Cell colors:
  - Light blue = healthy
  - Light green = dividing
  - Yellow = stressed
  - Gray = dead

### **Top Right: Single Cell Internal View** ⭐ ANIMATED!
- **Purple dots** = DNA (chromatin in nucleus)
- **Orange dots** = mRNA (moving from nucleus to cytoplasm!)
- **Green dots** = Ribosomes (scattered in cytoplasm)
- **Cyan dots** = Proteins (throughout cell)
- **Red circles** = Mitochondria (rotating around cell!)
- **Yellow rectangles** = Membrane receptors
- **Blue circle** = Nucleus

**Watch the mitochondria rotate!** 🔄

### **Bottom Left: Molecular Counts**
- Orange line = mRNA count over time
- Cyan line = Protein count over time
- Red line = ATP levels over time
- Shows real stochastic variation!

### **Bottom Center: Environment Gradients**
- Heat map of oxygen concentration
- Red = high oxygen
- Blue = low oxygen
- Black contour lines show gradients
- Cells consume oxygen → creates gradients

### **Bottom Right: Cell Signaling**
- Green = Growth factor signals
- Red = Stress signals
- Blue = Contact inhibition
- Shows cell-cell communication

---

## 🎯 Key Features

### 1. **Real-Time Simulation**
- Not a video or animation
- Actual simulation running live
- ~20 FPS (50ms per frame)
- 2 hours simulated time

### 2. **Animated Organelles**
```python
# Mitochondria rotate around cell
angle = 2 * π * i / n_mito + frame * 0.05
x = radius * cos(angle)
y = radius * sin(angle)
```

### 3. **mRNA Transport**
```python
# mRNA moves from nucleus (r=5) to cytoplasm (r=10)
progress = (frame * 0.1 + i * 0.5) % 1.0
r = 5 + (10 - 5) * progress
```

### 4. **Cell-Cell Interactions**
- Cells within 30μm interact
- Red dashed lines show connections
- Strength fades with distance

### 5. **Oxygen Diffusion**
```python
# Cells consume oxygen
grid.consume_field('oxygen', position, rate=0.1)

# Oxygen diffuses
grid.diffuse_field(oxygen, D=2000 μm²/s)
```

---

## 🎨 Customization

### Change Number of Cells

Edit `live_demo.py`:
```python
sim = Simulation(n_cells=10, dt=0.01)  # Change to 10 cells

# Add more positions
positions = [
    (20, 20, 50),
    (40, 20, 50),
    (60, 20, 50),
    # ... add more
]
```

### Change Simulation Speed

```python
create_live_simulation(
    duration_hours=5.0,    # Longer simulation
    dt=0.005,              # Smaller time steps (slower)
    interval_ms=100        # Slower animation (10 FPS)
)
```

### Change Animation Speed

```python
interval_ms=25   # Faster (40 FPS)
interval_ms=100  # Slower (10 FPS)
```

---

## 🔬 What's Being Simulated

### Internal Processes
1. **DNA Transcription**
   - Stochastic (Poisson process)
   - Creates mRNA molecules
   - Consumes ATP, GTP

2. **mRNA Translation**
   - Ribosomes bind to mRNA
   - Produce proteins
   - Consumes ATP, amino acids

3. **Protein Degradation**
   - Proteins decay over time
   - Half-life ~1 hour

4. **Metabolism**
   - ATP production (mitochondria)
   - ATP consumption (processes)

### External Processes
1. **Oxygen Diffusion**
   - PDE solver: ∂C/∂t = D∇²C
   - D = 2000 μm²/s
   - Numerical stability checked

2. **Cell Consumption**
   - Cells consume oxygen
   - Creates gradients
   - Affects cell health

3. **Cell-Cell Signaling**
   - Distance-based interactions
   - Signal strength fades
   - Affects cell behavior

---

## 📈 Performance

### Current
- **5 cells**: ~20 FPS (smooth!)
- **10 cells**: ~15 FPS
- **20 cells**: ~10 FPS

### Bottlenecks
1. Matplotlib rendering (main bottleneck)
2. Diffusion solver (secondary)
3. Drawing individual molecules

### Future Optimizations
- Use Plotly (WebGL) for 3D
- GPU acceleration (Phase 4)
- Reduce drawing frequency
- Use sprites for molecules

---

## 🎯 Use Cases

### 1. **Demonstrations**
- Show to collaborators
- Grant presentations
- Conference talks
- Teaching

### 2. **Development**
- Debug simulation
- Verify behavior
- Test new features
- Parameter tuning

### 3. **Exploration**
- Understand dynamics
- Discover patterns
- Generate hypotheses
- Visual intuition

---

## 🛠️ Technical Details

### Architecture

```python
LiveCellularVisualizer
├── 5 subplots (matplotlib)
├── FuncAnimation (real-time updates)
├── Simulation (backend)
└── SpatialGrid (environment)
```

### Update Loop

```python
def update_frame(frame):
    1. Run simulation step
    2. Update spatial grid
    3. Collect data
    4. Redraw all panels
    5. Update title/stats
```

### Data Flow

```
Simulation → Cells → Intracellular → Molecules
     ↓
SpatialGrid → Environment → Gradients
     ↓
Visualizer → Plots → Screen
```

---

## 🎨 Color Scheme

### Molecules
- **Purple** = DNA
- **Orange** = mRNA
- **Green** = Ribosomes
- **Cyan** = Proteins
- **Red** = Mitochondria
- **Yellow** = Receptors

### Cells
- **Light blue** = Healthy
- **Light green** = Dividing
- **Yellow** = Stressed
- **Gray** = Dead

### Environment
- **Blue** = High oxygen
- **Red** = Low oxygen

---

## 🚀 Next Steps

### Short Term (This Week)
- [ ] Add more cell types (cancer, immune)
- [ ] Show cell division animation
- [ ] Add cell death animation
- [ ] Show receptor activation

### Medium Term (This Month)
- [ ] 3D visualization (Plotly)
- [ ] Interactive controls (pause, speed)
- [ ] Save animation to video
- [ ] Add more signaling pathways

### Long Term (Phase 4+)
- [ ] GPU acceleration
- [ ] 100+ cells real-time
- [ ] VR/AR visualization
- [ ] Web-based viewer

---

## 💡 Tips

### For Best Performance
```bash
# Close other applications
# Use smaller window
# Reduce number of cells
# Increase interval_ms
```

### For Best Quality
```bash
# Use larger figsize
# Reduce interval_ms (slower but smoother)
# Save to video instead of live viewing
```

### For Debugging
```python
# Add print statements in update_frame()
print(f"Frame {frame}: {len(sim.cells)} cells")

# Check data
print(f"mRNA: {molecular_data['mrna'][-1]}")
```

---

## 🎬 Example Output

When you run `live_demo.py`, you'll see:

```
============================================================
cognisom: Live Cellular Simulation
============================================================

🎬 Starting interactive visualization...

What you'll see:
  • Top Left: Multiple cells in spatial environment
  • Top Right: Single cell internal view (animated!)
  • Bottom Left: Molecular counts over time
  • Bottom Center: Environment gradients (oxygen)
  • Bottom Right: Cell signaling activity

Features:
  ✓ Real-time simulation
  ✓ Animated organelles (mitochondria rotate!)
  ✓ mRNA transport (nucleus → cytoplasm)
  ✓ Cell-cell interactions
  ✓ Oxygen diffusion and consumption
  ✓ Live molecular counts

Press Ctrl+C to stop
============================================================

[Window opens with live animation]
```

---

## 🐛 Troubleshooting

### "No module named 'matplotlib'"
```bash
pip install matplotlib numpy
```

### Animation is slow
```python
# Increase interval
interval_ms=100  # 10 FPS instead of 20
```

### Window doesn't open
```bash
# Check matplotlib backend
python3 -c "import matplotlib; print(matplotlib.get_backend())"

# Should be: TkAgg or Qt5Agg
```

### Simulation crashes
```bash
# Reduce complexity
n_cells=3  # Fewer cells
duration_hours=1.0  # Shorter simulation
```

---

## 📚 Code Structure

### Main Files
- `engine/py/live_visualizer.py` - Visualization engine
- `live_demo.py` - Simple demo script
- `engine/py/simulation.py` - Simulation backend
- `engine/py/spatial/grid.py` - Spatial grid

### Key Classes
- `LiveCellularVisualizer` - Main visualizer
- `Simulation` - Simulation engine
- `SpatialGrid` - Environment
- `IntracellularModel` - Cell internals

---

## 🎉 Summary

**You now have**:
- ✅ Real-time interactive visualization
- ✅ Internal + external cellular dynamics
- ✅ Animated organelles and molecules
- ✅ Cell-cell interactions
- ✅ Environmental gradients
- ✅ Live molecular counts

**Just run**:
```bash
python3 live_demo.py
```

**And watch your cells come alive!** 🧬✨🎬

---

## 🚀 Advanced: Save to Video

Want to save the animation?

```python
from matplotlib.animation import FFMpegWriter

# In live_visualizer.py, after creating animation:
writer = FFMpegWriter(fps=20)
anim.save('simulation.mp4', writer=writer)
```

Requires `ffmpeg`:
```bash
brew install ffmpeg  # macOS
```

---

**Enjoy your live cellular simulation!** 🎬🧬💻
