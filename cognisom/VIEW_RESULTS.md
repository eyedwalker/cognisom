# 🖼️ How to View Your Results

## Quick Commands

### View Visualization (Fastest)
```bash
open output/basic_growth/simulation_results.png
```

### View Enhanced Visualization
```bash
python3 explore_results.py
```

### View Raw Data
```bash
cat output/basic_growth/results.json
```

### View Summary
```bash
python3 view_results.py
```

---

## Output Files Explained

### `simulation_results.png`
**4-panel visualization** showing:
1. **Top-left**: Population growth (cell count over time)
2. **Top-right**: Total protein content (biomass)
3. **Bottom-left**: MHC-I expression (immune marker)
4. **Bottom-right**: Cellular stress levels

### `enhanced_results.png`
**Improved version** with:
- Doubling time annotation
- Reference lines (normal/critical thresholds)
- Better colors and labels
- Higher resolution

### `results.json`
**Raw data** containing:
- Time points
- Cell counts at each time
- Protein levels
- MHC-I expression
- Stress levels
- Event counts (divisions, deaths)

---

## Interactive Viewing

### Option 1: Python Script
```bash
python3 explore_results.py
```
**Shows**:
- Detailed text summary
- Interactive matplotlib plot
- Saves enhanced visualization

### Option 2: Jupyter Notebook
```bash
jupyter notebook
# Create new notebook
# Load and explore results interactively
```

### Option 3: Custom Analysis
```python
import json
import matplotlib.pyplot as plt

# Load results
with open('output/basic_growth/results.json') as f:
    results = json.load(f)

# Plot whatever you want
plt.plot(results['history']['time'], 
         results['history']['cell_count'])
plt.show()
```

---

## Understanding the Plots

### Population Growth
- **X-axis**: Time in hours
- **Y-axis**: Number of cells
- **Expected**: Exponential growth (1 → 2 → 4 → 8...)
- **Your result**: 1 → 2 in 23 hours ✓

### Total Proteins
- **X-axis**: Time in hours
- **Y-axis**: Sum of all proteins in all cells
- **Expected**: Increases as cells grow
- **Pattern**: Dips at division (proteins split between cells)

### MHC-I Expression
- **X-axis**: Time in hours
- **Y-axis**: Average MHC-I level (0-1 scale)
- **Normal**: 1.0 (full expression)
- **Reduced**: <0.5 (immune evasion)
- **Your result**: 1.0 (healthy cells) ✓

### Cellular Stress
- **X-axis**: Time in hours
- **Y-axis**: Average stress level (0-1 scale)
- **Low**: 0-0.3 (normal)
- **Moderate**: 0.3-0.7 (stressed)
- **Critical**: >0.9 (triggers apoptosis)
- **Your result**: 0.0 (no stress) ✓

---

## Common Questions

### Q: Why does protein count decrease over time?
**A**: Protein degradation! Cells constantly break down proteins. The balance between synthesis and degradation determines net growth.

### Q: Why did the cell divide at 23 hours?
**A**: The cell cycle takes ~24 hours (G1: 11h, S: 8h, G2: 4h, M: 1h). Division happens when:
1. Cell reaches M phase
2. Protein count > threshold (2,500)

### Q: Is 23 hours realistic?
**A**: Yes! Mammalian cells typically divide every 18-36 hours. Your simulation is biologically accurate.

### Q: Why is MHC-I always 1.0?
**A**: No stress was applied. In future examples with stress/mutations, you'll see MHC-I decrease (immune evasion).

---

## Viewing Results from Different Simulations

### List all output directories
```bash
ls -la output/
```

### View specific simulation
```bash
open output/stress_response/simulation_results.png
cat output/stress_response/results.json
```

### Compare multiple runs
```python
import json
import matplotlib.pyplot as plt

# Load multiple results
runs = ['basic_growth', 'stress_response', 'multi_cell']
for run in runs:
    with open(f'output/{run}/results.json') as f:
        data = json.load(f)
    plt.plot(data['history']['time'], 
             data['history']['cell_count'], 
             label=run)

plt.legend()
plt.show()
```

---

## Export Options

### Save as PDF
```python
import matplotlib.pyplot as plt
from pathlib import Path

# Load and plot
# ... your plotting code ...

# Save as PDF
plt.savefig('output/results.pdf', format='pdf', dpi=300)
```

### Save as CSV
```python
import json
import pandas as pd

# Load results
with open('output/basic_growth/results.json') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results['history'])

# Save as CSV
df.to_csv('output/results.csv', index=False)
```

### Save as Excel
```python
df.to_excel('output/results.xlsx', index=False)
```

---

## Troubleshooting

### "File not found"
**Problem**: No results exist yet  
**Solution**: Run a simulation first
```bash
python3 examples/single_cell/basic_growth.py
```

### "matplotlib not installed"
**Problem**: Missing dependency  
**Solution**: Install it
```bash
pip3 install matplotlib
```

### "Image won't open"
**Problem**: `open` command doesn't work on your system  
**Solution**: Use alternative
```bash
# Linux
xdg-open output/basic_growth/simulation_results.png

# Windows
start output/basic_growth/simulation_results.png

# Or just navigate in file browser
```

---

## Quick Reference Card

| Task | Command |
|------|---------|
| View plot | `open output/basic_growth/simulation_results.png` |
| View data | `cat output/basic_growth/results.json` |
| Enhanced view | `python3 explore_results.py` |
| Summary | `python3 view_results.py` |
| List outputs | `ls -la output/` |
| Run new sim | `python3 examples/single_cell/basic_growth.py` |

---

## Next Steps

1. ✅ You've seen your results
2. ⏳ Run another simulation with different parameters
3. ⏳ Create a stress response example
4. ⏳ Compare multiple runs
5. ⏳ Export for presentations

---

**Your cells are growing. Your data is beautiful. Keep building!** 🧬📊🚀
