#!/usr/bin/env python3
"""
Live Interactive Cellular Simulation Demo
=========================================

Real-time visualization showing:
- Multiple cells in spatial environment
- Internal cellular processes (DNA, RNA, proteins)
- Cell-cell interactions
- Environmental gradients
- Molecular dynamics

Press Ctrl+C to stop.
"""

import sys
sys.path.insert(0, '.')

from cognisom.engine.py.live_visualizer import create_live_simulation

if __name__ == '__main__':
    print("=" * 60)
    print("cognisom: Live Cellular Simulation")
    print("=" * 60)
    print()
    print("🎬 Starting interactive visualization...")
    print()
    print("What you'll see:")
    print("  • Top Left: Multiple cells in spatial environment")
    print("  • Top Right: Single cell internal view (animated!)")
    print("  • Bottom Left: Molecular counts over time")
    print("  • Bottom Center: Environment gradients (oxygen)")
    print("  • Bottom Right: Cell signaling activity")
    print()
    print("Features:")
    print("  ✓ Real-time simulation")
    print("  ✓ Animated organelles (mitochondria rotate!)")
    print("  ✓ mRNA transport (nucleus → cytoplasm)")
    print("  ✓ Cell-cell interactions")
    print("  ✓ Oxygen diffusion and consumption")
    print("  ✓ Live molecular counts")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    try:
        # Run simulation
        # Duration: 2 hours simulated time
        # dt: 0.01 hours (36 seconds) per step
        # interval: 50ms between frames (20 FPS)
        anim, viz = create_live_simulation(
            duration_hours=2.0,
            dt=0.01,
            interval_ms=50
        )
    except KeyboardInterrupt:
        print("\n\n✓ Simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
