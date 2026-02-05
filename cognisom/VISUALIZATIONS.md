# 🖼️ Visualizations Location Guide

## 📁 Where Your Visualizations Are

### **All visualizations are in the `output/` directory**

```
cognisom/
└── output/
    ├── basic_growth/
    │   ├── simulation_results.png       (99 KB)  ← Basic 4-panel plot
    │   ├── enhanced_results.png         (135 KB) ← Enhanced version
    │   └── results.json                 (1.6 KB) ← Raw data
    │
    ├── intracellular/
    │   └── intracellular_dashboard.png  (237 KB) ← Cell internals
    │
    └── test_basic/
        └── (test outputs)
```

---

## 🎨 What Each Visualization Shows

### 1. **Intracellular Dashboard** (237 KB) ⭐ BEST
**Location**: `output/intracellular/intracellular_dashboard.png`

**Shows**:
- **Top-left**: Cell structure with all organelles
  - Nucleus with DNA
  - Mitochondria (8 orange ovals)
  - Endoplasmic Reticulum (green network)
  - Ribosomes (purple dots)
  - Golgi apparatus
  - Vesicles
  - Membrane receptors

- **Top-right**: Molecular dynamics over time
  - mRNA levels
  - Protein levels
  - ATP levels

- **Middle-right**: Gene expression (mRNA counts)
  - Which genes are active
  - Expression levels

- **Bottom**: Protein abundance
  - How many proteins of each type

**Open it**:
```bash
open output/intracellular/intracellular_dashboard.png
```

---

### 2. **Basic Growth Results** (99 KB)
**Location**: `output/basic_growth/simulation_results.png`

**Shows** (4 panels):
- **Top-left**: Population growth (1 → 2 cells)
- **Top-right**: Total protein content
- **Bottom-left**: MHC-I expression (immune marker)
- **Bottom-right**: Cellular stress levels

**Open it**:
```bash
open output/basic_growth/simulation_results.png
```

---

### 3. **Enhanced Results** (135 KB)
**Location**: `output/basic_growth/enhanced_results.png`

**Shows**: Same as basic but with:
- Better annotations
- Doubling time marked
- Reference lines
- Higher quality

**Open it**:
```bash
open output/basic_growth/enhanced_results.png
```

---

## 🚀 Quick Commands

### Open All Visualizations
```bash
# Open all at once
open output/intracellular/intracellular_dashboard.png
open output/basic_growth/simulation_results.png
open output/basic_growth/enhanced_results.png
```

### Open in Finder
```bash
# Open output folder in Finder
open output/
```

### List All Images
```bash
# See all visualizations
find output -name "*.png" -type f
```

---

## 🎨 Generate New Visualizations

### Regenerate Intracellular Dashboard
```bash
python3 examples/single_cell/intracellular_detail.py
# Creates: output/intracellular/intracellular_dashboard.png
```

### Regenerate Basic Growth
```bash
python3 examples/single_cell/basic_growth.py
# Creates: output/basic_growth/simulation_results.png
```

### Create Enhanced Version
```bash
python3 explore_results.py
# Creates: output/basic_growth/enhanced_results.png
```

---

## 📊 File Sizes

| File | Size | Description |
|------|------|-------------|
| `intracellular_dashboard.png` | 237 KB | Full cell internals (best!) |
| `enhanced_results.png` | 135 KB | Enhanced growth plots |
| `simulation_results.png` | 99 KB | Basic growth plots |

---

## 🖼️ What's in Each Visualization

### Intracellular Dashboard (The Best One!)
```
┌─────────────────────────────────────────────────────┐
│  Cell Structure          │  Molecular Dynamics      │
│  (with organelles)       │  (mRNA, Protein, ATP)    │
│                          │                          │
│  • Nucleus with DNA      │  [Time series plot]      │
│  • Mitochondria          │                          │
│  • ER, Golgi             ├──────────────────────────┤
│  • Ribosomes             │  Gene Expression         │
│  • Membrane receptors    │  (mRNA counts)           │
│                          │                          │
│                          │  [Bar chart]             │
├──────────────────────────┴──────────────────────────┤
│              Protein Abundance                      │
│              (horizontal bar chart)                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Basic Growth Results
```
┌──────────────────────┬──────────────────────┐
│  Population Growth   │  Total Proteins      │
│  (cells over time)   │  (biomass)           │
│                      │                      │
│  [Line plot]         │  [Line plot]         │
│                      │                      │
├──────────────────────┼──────────────────────┤
│  MHC-I Expression    │  Cellular Stress     │
│  (immune marker)     │  (stress level)      │
│                      │                      │
│  [Line plot]         │  [Line plot]         │
│                      │                      │
└──────────────────────┴──────────────────────┘
```

---

## 💡 Tips

### View in Browser
```bash
# If images don't open, try:
python3 -m http.server 8000
# Then go to: http://localhost:8000/output/
```

### Convert to PDF
```bash
# For presentations
convert output/intracellular/intracellular_dashboard.png dashboard.pdf
```

### Share
```bash
# Copy to desktop
cp output/intracellular/intracellular_dashboard.png ~/Desktop/
```

---

## 🎯 Which One to Look At?

**For understanding cell internals**:
→ `output/intracellular/intracellular_dashboard.png` ⭐

**For growth dynamics**:
→ `output/basic_growth/enhanced_results.png`

**For quick check**:
→ `output/basic_growth/simulation_results.png`

---

## 🔄 Regenerate All

```bash
# Run all examples to regenerate everything
python3 examples/single_cell/basic_growth.py
python3 examples/single_cell/intracellular_detail.py
python3 explore_results.py

# All visualizations updated!
```

---

## 📍 Absolute Paths

If you need full paths:
```bash
# Get absolute paths
pwd
# Then add: /output/intracellular/intracellular_dashboard.png
```

Your visualizations are at:
```
/Users/davidwalker/CascadeProjects/cognisom/output/intracellular/intracellular_dashboard.png
/Users/davidwalker/CascadeProjects/cognisom/output/basic_growth/simulation_results.png
/Users/davidwalker/CascadeProjects/cognisom/output/basic_growth/enhanced_results.png
```

---

## 🎉 Summary

**Location**: `output/` directory  
**Count**: 3 visualizations  
**Total size**: 471 KB  
**Best one**: `intracellular_dashboard.png` (shows everything!)  

**Open them all**:
```bash
open output/intracellular/intracellular_dashboard.png
open output/basic_growth/enhanced_results.png
```

---

**Your visualizations are ready and beautiful!** 🎨✨
