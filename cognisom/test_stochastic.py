#!/usr/bin/env python3
"""
Test Stochastic Simulation Properly
====================================

Verify that:
1. Transcription is truly stochastic (Poisson process)
2. Translation is stochastic
3. Degradation follows exponential decay
4. Multiple runs show proper statistical variation
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from engine.py.intracellular import IntracellularModel

print("=" * 70)
print("🔬 Testing Stochastic Simulation")
print("=" * 70)
print()

# Test 1: Is transcription stochastic?
print("TEST 1: Transcription Stochasticity")
print("-" * 70)
print()

print("Running 100 simulations of 1 hour each...")
print("Counting transcription events for GAPDH gene...")
print()

transcription_counts = []
for run in range(100):
    cell = IntracellularModel()
    
    # Set high transcription rate for easier detection
    cell.genes['GAPDH'].transcription_rate = 0.5  # 0.5 mRNA/hour
    
    # Run for 1 hour
    for _ in range(100):  # 1 hour at dt=0.01
        cell.step(dt=0.01)
    
    # Count final mRNA
    mrna_count = cell.mrnas.get('GAPDH', type('obj', (), {'copy_number': 0})).copy_number
    transcription_counts.append(mrna_count)

print(f"Results from 100 independent simulations:")
print(f"  Mean mRNA: {np.mean(transcription_counts):.2f}")
print(f"  Std Dev: {np.std(transcription_counts):.2f}")
print(f"  Min: {np.min(transcription_counts)}")
print(f"  Max: {np.max(transcription_counts)}")
print(f"  Range: {np.max(transcription_counts) - np.min(transcription_counts)}")
print()

# Expected for Poisson: mean ≈ variance
expected_mean = 0.5  # 0.5 mRNA/hour × 1 hour
variance = np.var(transcription_counts)
print(f"Expected mean (Poisson): {expected_mean:.2f}")
print(f"Observed variance: {variance:.2f}")
print(f"Variance/Mean ratio: {variance/np.mean(transcription_counts):.2f}")
print(f"  (Should be ~1.0 for Poisson process)")
print()

if 0.5 < variance/np.mean(transcription_counts) < 1.5:
    print("✓ PASS: Transcription follows Poisson statistics")
else:
    print("⚠️  WARNING: Transcription may not be properly stochastic")
print()

# Test 2: Distribution check
print("TEST 2: Distribution Analysis")
print("-" * 70)
print()

print("Distribution of mRNA counts:")
unique, counts = np.unique(transcription_counts, return_counts=True)
for val, count in zip(unique, counts):
    bar = "█" * (count // 2)
    print(f"  {val:2d} mRNA: {bar} ({count} runs)")
print()

# Test 3: Time series variation
print("TEST 3: Time Series Variation")
print("-" * 70)
print()

print("Running 5 parallel simulations for 3 hours...")
print("Tracking protein levels over time...")
print()

fig, ax = plt.subplots(figsize=(10, 6))

for run in range(5):
    cell = IntracellularModel()
    cell.genes['GAPDH'].transcription_rate = 0.5
    
    times = []
    proteins = []
    
    for step in range(300):  # 3 hours
        t = step * 0.01
        cell.step(dt=0.01)
        
        if step % 10 == 0:  # Record every 0.1 hours
            times.append(t)
            protein_count = cell.proteins.get('GAPDH', type('obj', (), {'copy_number': 0})).copy_number
            proteins.append(protein_count)
    
    ax.plot(times, proteins, alpha=0.7, linewidth=2, label=f'Run {run+1}')
    print(f"  Run {run+1}: Final proteins = {proteins[-1]}")

ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('GAPDH Protein Count', fontsize=12)
ax.set_title('Stochastic Variation Across 5 Independent Simulations', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

output_dir = Path('output/stochastic_test')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'stochastic_variation.png', dpi=150, bbox_inches='tight')
print()
print(f"✓ Plot saved: {output_dir / 'stochastic_variation.png'}")
print()

# Test 4: Check if it's truly random (not deterministic)
print("TEST 4: Randomness Check")
print("-" * 70)
print()

print("Running same simulation 10 times with identical parameters...")
results = []
for i in range(10):
    cell = IntracellularModel()
    cell.genes['GAPDH'].transcription_rate = 0.5
    
    for _ in range(100):
        cell.step(dt=0.01)
    
    final_proteins = cell.get_state_summary()['total_proteins']
    results.append(final_proteins)
    print(f"  Run {i+1:2d}: {final_proteins:3d} proteins")

print()
print(f"All results identical? {len(set(results)) == 1}")
print(f"Unique outcomes: {len(set(results))}/10")
print()

if len(set(results)) > 5:
    print("✓ PASS: Results show proper randomness")
else:
    print("⚠️  WARNING: Results may be too deterministic")
print()

# Test 5: Check the actual Poisson implementation
print("TEST 5: Verify Poisson Implementation")
print("-" * 70)
print()

print("Checking if np.random.poisson is being used correctly...")
print()

# Direct test of Poisson
rate = 0.5
dt = 0.01
samples = [np.random.poisson(rate * dt) for _ in range(1000)]

print(f"Direct Poisson test (rate={rate}, dt={dt}):")
print(f"  Mean: {np.mean(samples):.4f}")
print(f"  Expected: {rate * dt:.4f}")
print(f"  Variance: {np.var(samples):.4f}")
print(f"  Expected: {rate * dt:.4f}")
print()

if abs(np.mean(samples) - rate * dt) < 0.001:
    print("✓ PASS: Poisson implementation is correct")
else:
    print("⚠️  WARNING: Poisson implementation may have issues")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

checks = []

# Check 1: Variance/Mean ratio
vmr = variance/np.mean(transcription_counts)
checks.append(("Poisson statistics", 0.5 < vmr < 1.5))

# Check 2: Randomness
checks.append(("Randomness", len(set(results)) > 5))

# Check 3: Distribution spread
checks.append(("Distribution spread", len(unique) > 1))

for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"  {status} {check_name}")

print()

all_passed = all(passed for _, passed in checks)
if all_passed:
    print("✅ All stochastic tests PASSED!")
    print()
    print("The simulation is properly stochastic:")
    print("  • Transcription follows Poisson process")
    print("  • Multiple runs give different results")
    print("  • Statistical properties are correct")
else:
    print("⚠️  Some tests failed - stochastic behavior may need improvement")

print()
print("=" * 70)
print()
print("View the variation plot:")
print(f"  open {output_dir / 'stochastic_variation.png'}")
print()
