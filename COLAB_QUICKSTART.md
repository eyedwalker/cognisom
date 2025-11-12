# 🚀 Google Colab Quick Start (No Clone Needed!)

## Problem: Repository is Private

Since the repo might be private, here are **3 ways** to run cognisom on Colab:

---

## ✅ Option 1: Make Repo Public (Easiest!)

### **On GitHub**:
1. Go to: https://github.com/eyedwalker/cognisom
2. Click **Settings**
3. Scroll to **Danger Zone**
4. Click **Change visibility**
5. Select **Public**
6. Confirm

### **Then in Colab**:
```python
!git clone https://github.com/eyedwalker/cognisom.git
%cd cognisom
!pip install -q numpy scipy matplotlib flask flask-cors
!python3 test_platform.py
```

---

## ✅ Option 2: Upload Files to Colab (Works Now!)

### **Step 1: Download from GitHub**
1. Go to: https://github.com/eyedwalker/cognisom
2. Click **Code** → **Download ZIP**
3. Extract ZIP on your computer

### **Step 2: Upload to Colab**
```python
# In Colab, run this first:
!pip install -q numpy scipy matplotlib flask flask-cors

# Then upload files:
# 1. Click folder icon on left
# 2. Click upload button
# 3. Upload entire cognisom folder
```

### **Step 3: Run**
```python
%cd cognisom
!python3 test_platform.py
```

---

## ✅ Option 3: Direct Code (Copy/Paste)

### **Run this entire code block in Colab**:

```python
# Install dependencies
!pip install -q numpy scipy matplotlib flask flask-cors

# Create minimal working version
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

print("🧬 cognisom Mini Demo")
print("=" * 50)

# Minimal simulation
class MiniSimulation:
    def __init__(self):
        self.time = 0.0
        self.cancer_cells = 10
        self.normal_cells = 80
        self.immune_cells = 20
        self.immune_kills = 0
    
    def step(self, dt=0.01):
        """Run one simulation step"""
        self.time += dt
        
        # Simple immune-cancer interaction
        if self.cancer_cells > 0 and self.immune_cells > 0:
            # Immune cells kill cancer
            kill_prob = 0.001 * self.immune_cells * dt
            if np.random.random() < kill_prob:
                self.cancer_cells -= 1
                self.immune_kills += 1
        
        # Cancer cells divide
        if self.cancer_cells > 0:
            division_prob = 0.01 * dt
            if np.random.random() < division_prob:
                self.cancer_cells += 1
    
    def run(self, duration=1.0):
        """Run simulation"""
        steps = int(duration / 0.01)
        for _ in range(steps):
            self.step()
    
    def get_results(self):
        """Get results"""
        return {
            'time': self.time,
            'cancer_cells': self.cancer_cells,
            'normal_cells': self.normal_cells,
            'immune_kills': self.immune_kills
        }

# Run simulation
sim = MiniSimulation()
print(f"Initial: {sim.cancer_cells} cancer cells")

print("\nRunning 10-hour simulation...")
sim.run(duration=10.0)

results = sim.get_results()
print(f"\n✅ Complete!")
print(f"Time: {results['time']:.1f}h")
print(f"Cancer cells: {results['cancer_cells']}")
print(f"Immune kills: {results['immune_kills']}")

# Run multiple scenarios
print("\n" + "=" * 50)
print("SCENARIO: Boost Immune System")
print("=" * 50)

sim2 = MiniSimulation()
sim2.immune_cells = 50  # 2.5x boost!
print(f"Initial: {sim2.cancer_cells} cancer cells, {sim2.immune_cells} immune cells")

sim2.run(duration=10.0)
results2 = sim2.get_results()

print(f"\n✅ Complete!")
print(f"Cancer cells: {results2['cancer_cells']}")
print(f"Immune kills: {results2['immune_kills']}")

print("\n🎉 cognisom mini demo complete!")
print("This is a simplified version. Upload full code for all features!")
```

---

## ✅ Option 4: Use Google Drive

### **Step 1: Upload to Drive**
1. Upload cognisom folder to Google Drive
2. Share folder (get link)

### **Step 2: Mount Drive in Colab**
```python
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your folder
%cd /content/drive/MyDrive/cognisom

# Install dependencies
!pip install -q numpy scipy matplotlib flask flask-cors

# Run
!python3 test_platform.py
```

---

## 🎯 Recommended Approach

### **For Quick Testing** (Right Now!):
**Use Option 3** - Copy/paste the mini demo above
- Works immediately
- No files needed
- Shows the concept

### **For Full Features**:
**Use Option 1** - Make repo public
- One-time setup
- Full functionality
- Easy to update

### **For Private Repo**:
**Use Option 2** - Upload files
- Keeps repo private
- Full functionality
- Manual updates

---

## 📝 Full Code for Colab (Copy/Paste)

If you want to run the full platform without cloning:

### **Step 1: Install**
```python
!pip install -q numpy scipy matplotlib flask flask-cors
```

### **Step 2: Create Files**

You'll need to upload these folders:
- `core/` - Core engine
- `modules/` - All 9 modules
- `scenarios/` - Pre-built scenarios
- `api/` - REST API

### **Step 3: Run**
```python
# After uploading files:
!python3 test_platform.py
```

---

## 🆘 Troubleshooting

### **Error: "fatal: could not read Username"**
→ Repo is private. Use Option 1 (make public) or Option 2 (upload files)

### **Error: "No such file or directory"**
→ Files not uploaded. Use Option 2 or 3

### **Error: "ModuleNotFoundError"**
→ Run: `!pip install numpy scipy matplotlib flask flask-cors`

---

## ✅ What Works Right Now

**Copy this into Colab and run**:

```python
# Quick demo that works immediately
!pip install -q numpy scipy matplotlib

import numpy as np

print("🧬 cognisom Quick Demo")
print("Simulating cancer vs immune cells...")

cancer = 10
immune = 20
kills = 0

for hour in range(24):
    # Immune kills cancer
    if cancer > 0 and np.random.random() < 0.1:
        cancer -= 1
        kills += 1
    
    # Cancer divides
    if cancer > 0 and np.random.random() < 0.05:
        cancer += 1
    
    if hour % 6 == 0:
        print(f"Hour {hour}: {cancer} cancer cells, {kills} kills")

print(f"\n✅ Final: {cancer} cancer cells, {kills} total kills")
```

---

## 🎯 Summary

**Easiest** (works now):
- Copy/paste Option 3 code above

**Best** (full features):
- Make repo public (Option 1)
- Or upload files (Option 2)

**The mini demo works RIGHT NOW - try it!** 🚀
